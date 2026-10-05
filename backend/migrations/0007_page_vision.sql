-- Per-page decision whether model stages must see the page image (issue #38).
-- Text-only pages are embedded, reranked, and answered from their text layer.
--
-- Existing rows were embedded from page images, so they keep needs_vision =
-- true until the document is re-ingested. The temporary defaults only backfill
-- them; every later write must state the value explicitly.

ALTER TABLE page_artifacts
  ADD COLUMN needs_vision boolean NOT NULL DEFAULT true;
ALTER TABLE page_artifacts
  ALTER COLUMN needs_vision DROP DEFAULT;
ALTER TABLE page_artifacts
  ADD CONSTRAINT page_artifacts_text_only_has_text_check
  CHECK (needs_vision OR btrim(coalesce(extracted_text, '')) <> '');

ALTER TABLE document_units
  ADD COLUMN needs_vision boolean NOT NULL DEFAULT true;
-- Only the new column changes, so the retrieval-ready and BM25 statistics
-- triggers (UPDATE OF document_id, retrieval_ready, content) do not fire.
UPDATE document_units
SET needs_vision = false
WHERE kind = 'text';
ALTER TABLE document_units
  ALTER COLUMN needs_vision DROP DEFAULT;
ALTER TABLE document_units
  ADD CONSTRAINT document_units_vision_kind_check
  CHECK (kind = 'page' OR NOT needs_vision);
ALTER TABLE document_units
  ADD CONSTRAINT document_units_text_only_has_content_check
  CHECK (needs_vision OR btrim(coalesce(content, '')) <> '');
