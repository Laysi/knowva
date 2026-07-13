-- Keep previously-applied migration 0003 immutable. Binary originals such as
-- Outlook MSG files are stored and indexed through a metadata text unit.
ALTER TABLE document_artifact_sets
  DROP CONSTRAINT document_artifact_sets_source_kind_check;

ALTER TABLE document_artifact_sets
  ADD CONSTRAINT document_artifact_sets_source_kind_check
    CHECK (source_kind IN ('pdf', 'image', 'office', 'text', 'binary'));
