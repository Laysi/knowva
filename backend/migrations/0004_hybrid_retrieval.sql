-- Hybrid retrieval keeps both ranking paths inside PostgreSQL:
--
-- * halfvec(2048) cosine search uses a ready-only HNSW index;
-- * lexical search uses an NFKC-normalized token array plus a ready-only GIN
--   inverted index and the Okapi BM25 formula in the application query.
--
-- PostgreSQL's ts_rank/ts_rank_cd are intentionally not used: neither is BM25.
-- The small statistics tables below are transactionally maintained from the
-- same rows as the GIN index, so BM25 does not rescan the whole corpus for N,
-- average document length, or document frequency on every query.

CREATE FUNCTION km_bm25_tokenize(input text)
RETURNS text[]
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
  normalized text;
  tokens text[] := ARRAY[]::text[];
  ascii_token text := '';
  compact_token text;
  component text;
  previous_cjk text := NULL;
  character text;
  character_index integer;
BEGIN
  IF input IS NULL OR btrim(input) = '' THEN
    RETURN tokens;
  END IF;

  -- NFKC makes full-width Latin letters/numbers/punctuation comparable with
  -- their half-width forms. Normalize the common Unicode dash variants too,
  -- because product identifiers frequently arrive from PDF text extraction.
  normalized := lower(normalize(input, NFKC));
  normalized := translate(normalized, '‐‑‒–—―﹘﹣－', '---------');

  FOR character_index IN 1..char_length(normalized) LOOP
    character := substr(normalized, character_index, 1);

    IF character ~ '^[a-z0-9]$'
       OR (character = '-' AND ascii_token <> '') THEN
      ascii_token := ascii_token || character;
      previous_cjk := NULL;
      CONTINUE;
    END IF;

    IF ascii_token <> '' THEN
      ascii_token := trim(BOTH '-' FROM ascii_token);
      IF ascii_token <> '' THEN
        tokens := array_append(tokens, ascii_token);
        compact_token := replace(ascii_token, '-', '');
        IF compact_token <> '' AND compact_token <> ascii_token THEN
          tokens := array_append(tokens, compact_token);
          FOREACH component IN ARRAY string_to_array(ascii_token, '-') LOOP
            IF component <> '' AND component <> ascii_token THEN
              tokens := array_append(tokens, component);
            END IF;
          END LOOP;
        END IF;
      END IF;
      ascii_token := '';
    END IF;

    -- Character unigrams make short Traditional Chinese queries reliable;
    -- adjacent bigrams add useful word-level discrimination without requiring
    -- a mutable external dictionary. Punctuation resets bigram adjacency.
    IF character ~ '^[㐀-䶿一-鿿豈-﫿]$' THEN
      tokens := array_append(tokens, character);
      IF previous_cjk IS NOT NULL THEN
        tokens := array_append(tokens, previous_cjk || character);
      END IF;
      previous_cjk := character;
    ELSE
      previous_cjk := NULL;
    END IF;
  END LOOP;

  IF ascii_token <> '' THEN
    ascii_token := trim(BOTH '-' FROM ascii_token);
    IF ascii_token <> '' THEN
      tokens := array_append(tokens, ascii_token);
      compact_token := replace(ascii_token, '-', '');
      IF compact_token <> '' AND compact_token <> ascii_token THEN
        tokens := array_append(tokens, compact_token);
        FOREACH component IN ARRAY string_to_array(ascii_token, '-') LOOP
          IF component <> '' AND component <> ascii_token THEN
            tokens := array_append(tokens, component);
          END IF;
        END LOOP;
      END IF;
    END IF;
  END IF;

  RETURN tokens;
END;
$$;

ALTER TABLE document_units
  ADD COLUMN retrieval_ready boolean NOT NULL DEFAULT false,
  ADD COLUMN bm25_tokens text[]
    GENERATED ALWAYS AS (km_bm25_tokenize(content)) STORED,
  ADD COLUMN bm25_length integer
    GENERATED ALWAYS AS (cardinality(km_bm25_tokenize(content))) STORED;

-- Denormalizing readiness onto the indexed row lets both partial indexes avoid
-- inactive units. Every retrieval query also joins documents and checks the
-- canonical status as defense in depth.
UPDATE document_units AS unit
SET retrieval_ready = (document.status = 'ready')
FROM documents AS document
WHERE document.id = unit.document_id;

CREATE FUNCTION km_set_unit_retrieval_ready()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  SELECT document.status = 'ready'
  INTO NEW.retrieval_ready
  FROM documents AS document
  WHERE document.id = NEW.document_id;

  IF NOT FOUND THEN
    NEW.retrieval_ready := false;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER document_units_set_retrieval_ready
BEFORE INSERT OR UPDATE OF document_id, retrieval_ready ON document_units
FOR EACH ROW
EXECUTE FUNCTION km_set_unit_retrieval_ready();

CREATE FUNCTION km_sync_document_retrieval_ready()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF OLD.status IS DISTINCT FROM NEW.status THEN
    UPDATE document_units
    SET retrieval_ready = (NEW.status = 'ready')
    WHERE document_id = NEW.id
      AND retrieval_ready IS DISTINCT FROM (NEW.status = 'ready');
  END IF;
  RETURN NULL;
END;
$$;

CREATE TRIGGER documents_sync_retrieval_ready
AFTER UPDATE OF status ON documents
FOR EACH ROW
EXECUTE FUNCTION km_sync_document_retrieval_ready();

DROP INDEX document_units_embedding_hnsw;

CREATE INDEX document_units_embedding_hnsw
  ON document_units USING hnsw (embedding halfvec_cosine_ops)
  WHERE retrieval_ready;

CREATE INDEX document_units_bm25_tokens_gin
  ON document_units USING gin (bm25_tokens)
  WHERE retrieval_ready
    AND content IS NOT NULL
    AND btrim(content) <> '';

CREATE TABLE km_bm25_corpus_stats (
  singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
  document_count bigint NOT NULL CHECK (document_count >= 0),
  total_length bigint NOT NULL CHECK (total_length >= 0)
);

CREATE TABLE km_bm25_term_stats (
  term text PRIMARY KEY,
  document_frequency bigint NOT NULL CHECK (document_frequency > 0)
);

INSERT INTO km_bm25_corpus_stats (singleton, document_count, total_length)
SELECT true,
       count(*),
       COALESCE(sum(bm25_length), 0)
FROM document_units
WHERE retrieval_ready
  AND content IS NOT NULL
  AND btrim(content) <> '';

INSERT INTO km_bm25_term_stats (term, document_frequency)
SELECT token.term, count(*)
FROM document_units AS unit
CROSS JOIN LATERAL (
  SELECT DISTINCT value AS term
  FROM unnest(unit.bm25_tokens) AS value
) AS token
WHERE unit.retrieval_ready
  AND unit.content IS NOT NULL
  AND btrim(unit.content) <> ''
GROUP BY token.term;

CREATE FUNCTION km_bm25_add_stats(tokens text[], document_length integer)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE km_bm25_corpus_stats
  SET document_count = document_count + 1,
      total_length = total_length + document_length
  WHERE singleton;

  INSERT INTO km_bm25_term_stats (term, document_frequency)
  SELECT DISTINCT token, 1
  FROM unnest(tokens) AS token
  WHERE token <> ''
  ON CONFLICT (term) DO UPDATE
  SET document_frequency = km_bm25_term_stats.document_frequency + 1;
END;
$$;

CREATE FUNCTION km_bm25_remove_stats(tokens text[], document_length integer)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  UPDATE km_bm25_corpus_stats
  SET document_count = document_count - 1,
      total_length = total_length - document_length
  WHERE singleton;

  DELETE FROM km_bm25_term_stats AS stats
  WHERE stats.document_frequency = 1
    AND stats.term IN (
    SELECT DISTINCT token
    FROM unnest(tokens) AS token
    WHERE token <> ''
  );

  UPDATE km_bm25_term_stats AS stats
  SET document_frequency = stats.document_frequency - 1
  WHERE stats.term IN (
    SELECT DISTINCT token
    FROM unnest(tokens) AS token
    WHERE token <> ''
  );
END;
$$;

CREATE FUNCTION km_bm25_sync_stats()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  old_is_indexable boolean := false;
  new_is_indexable boolean := false;
BEGIN
  IF TG_OP <> 'INSERT' THEN
    old_is_indexable := OLD.retrieval_ready
      AND OLD.content IS NOT NULL
      AND btrim(OLD.content) <> '';
  END IF;

  IF TG_OP <> 'DELETE' THEN
    new_is_indexable := NEW.retrieval_ready
      AND NEW.content IS NOT NULL
      AND btrim(NEW.content) <> '';
  END IF;

  IF old_is_indexable THEN
    PERFORM km_bm25_remove_stats(OLD.bm25_tokens, OLD.bm25_length);
  END IF;
  IF new_is_indexable THEN
    PERFORM km_bm25_add_stats(NEW.bm25_tokens, NEW.bm25_length);
  END IF;

  RETURN NULL;
END;
$$;

CREATE TRIGGER document_units_sync_bm25_stats
AFTER INSERT OR DELETE OR UPDATE OF document_id, retrieval_ready, content ON document_units
FOR EACH ROW
EXECUTE FUNCTION km_bm25_sync_stats();
