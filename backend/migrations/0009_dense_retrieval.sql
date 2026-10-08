-- Retrieval now ranks dense candidates only and reranks them (issue #60):
-- no question in the measured sets needed BM25 to reach its source, while RRF
-- pushed dense-only finds (image-only pages, English documents asked in
-- Chinese) out of the rerank pool. Drop everything migration 0004 added for
-- the lexical route.
--
-- retrieval_ready, its two triggers and the ready-only HNSW index stay: dense
-- retrieval still filters on them. Dropping the generated columns only marks
-- them dropped, so this does not rewrite document_units.

DROP TRIGGER document_units_sync_bm25_stats ON document_units;
DROP FUNCTION knowva_bm25_sync_stats();
DROP FUNCTION knowva_bm25_add_stats(text[], integer);
DROP FUNCTION knowva_bm25_remove_stats(text[], integer);
DROP TABLE knowva_bm25_term_stats;
DROP TABLE knowva_bm25_corpus_stats;

DROP INDEX document_units_bm25_tokens_gin;
ALTER TABLE document_units
  DROP COLUMN bm25_length,
  DROP COLUMN bm25_tokens;
DROP FUNCTION knowva_bm25_tokenize(text);
