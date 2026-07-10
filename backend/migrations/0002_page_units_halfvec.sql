-- Legacy chunks contain 1024-dimensional embeddings from a different model space.
-- They must never be cast or mixed with the new 2048-dimensional VL embeddings.
UPDATE documents AS d
SET status = 'failed',
    error = 'Re-ingestion required: migration 0002 invalidated legacy 1024-dimensional embeddings.'
WHERE d.status = 'ready'
   OR EXISTS (
       SELECT 1
       FROM chunks AS c
       WHERE c.document_id = d.id
   );

DROP TABLE chunks;

ALTER TABLE documents
  ADD CONSTRAINT documents_status_check
  CHECK (status IN ('processing', 'ready', 'failed'));

CREATE TABLE document_units (
  id          bigserial PRIMARY KEY,
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  kind        text NOT NULL,
  unit_index  integer NOT NULL,
  page_number integer,
  content     text,
  image_path  text,
  embedding   halfvec(2048) NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT document_units_kind_check
    CHECK (kind IN ('page', 'text')),
  CONSTRAINT document_units_unit_index_check
    CHECK (unit_index >= 0),
  CONSTRAINT document_units_page_number_check
    CHECK (page_number IS NULL OR page_number >= 1),
  CONSTRAINT document_units_kind_fields_check
    CHECK (
      (
        kind = 'page'
        AND page_number IS NOT NULL
        AND image_path IS NOT NULL
        AND btrim(image_path) <> ''
      )
      OR
      (
        kind = 'text'
        AND page_number IS NULL
        AND content IS NOT NULL
        AND btrim(content) <> ''
        AND image_path IS NULL
      )
    ),
  CONSTRAINT document_units_document_kind_index_key
    UNIQUE (document_id, kind, unit_index)
);

CREATE UNIQUE INDEX document_units_document_page_number_key
  ON document_units (document_id, page_number)
  WHERE kind = 'page';

CREATE INDEX document_units_embedding_hnsw
  ON document_units USING hnsw (embedding halfvec_cosine_ops);
