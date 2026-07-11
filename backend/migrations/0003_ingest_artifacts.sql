-- Conversion artifacts deliberately live outside document_units. Issue #4 does
-- not call the embedding model, while document_units.embedding must remain a
-- real 2048-dimensional halfvec. Issue #5 promotes these staging rows into
-- document_units only after embedding succeeds.
ALTER TABLE documents
  ADD COLUMN artifact_status text NOT NULL DEFAULT 'pending',
  ADD CONSTRAINT documents_artifact_status_check
    CHECK (artifact_status IN ('pending', 'processing', 'ready', 'failed'));

CREATE TABLE document_artifact_sets (
  document_id       uuid PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  source_kind       text NOT NULL,
  artifact_dir      text NOT NULL,
  manifest_path     text NOT NULL,
  converted_pdf_path text,
  created_at        timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT document_artifact_sets_source_kind_check
    CHECK (source_kind IN ('pdf', 'image', 'office', 'text')),
  CONSTRAINT document_artifact_sets_paths_check
    CHECK (
      btrim(artifact_dir) <> ''
      AND btrim(manifest_path) <> ''
      AND (converted_pdf_path IS NULL OR btrim(converted_pdf_path) <> '')
    ),
  CONSTRAINT document_artifact_sets_office_pdf_check
    CHECK (
      (source_kind = 'office' AND converted_pdf_path IS NOT NULL)
      OR (source_kind <> 'office' AND converted_pdf_path IS NULL)
    )
);

CREATE TABLE page_artifacts (
  id             bigserial PRIMARY KEY,
  document_id    uuid NOT NULL REFERENCES document_artifact_sets(document_id) ON DELETE CASCADE,
  page_number    integer NOT NULL,
  image_path     text NOT NULL,
  width          integer NOT NULL,
  height         integer NOT NULL,
  extracted_text text,
  created_at     timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT page_artifacts_page_number_check CHECK (page_number >= 1),
  CONSTRAINT page_artifacts_image_path_check CHECK (btrim(image_path) <> ''),
  CONSTRAINT page_artifacts_dimensions_check CHECK (width > 0 AND height > 0),
  CONSTRAINT page_artifacts_document_page_key UNIQUE (document_id, page_number)
);

CREATE TABLE text_artifacts (
  id          bigserial PRIMARY KEY,
  document_id uuid NOT NULL REFERENCES document_artifact_sets(document_id) ON DELETE CASCADE,
  unit_index  integer NOT NULL,
  content     text NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT text_artifacts_unit_index_check CHECK (unit_index >= 0),
  CONSTRAINT text_artifacts_content_check CHECK (btrim(content) <> ''),
  CONSTRAINT text_artifacts_document_index_key UNIQUE (document_id, unit_index)
);
