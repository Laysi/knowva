-- The storage backend is chosen once, on the first start against an empty
-- database. Document tables store keys relative to that backend, so the
-- backend (and, for S3, bucket and prefix) must never change afterwards.
-- FILES_DIR and the S3 endpoint are intentionally not recorded.
CREATE TABLE storage_settings (
  singleton  boolean PRIMARY KEY DEFAULT true CHECK (singleton),
  backend    text NOT NULL,
  s3_bucket  text,
  s3_prefix  text,
  created_at timestamptz NOT NULL DEFAULT now(),

  CONSTRAINT storage_settings_backend_check CHECK (
    (backend = 'local' AND s3_bucket IS NULL AND s3_prefix IS NULL)
    OR (backend = 's3' AND s3_bucket IS NOT NULL AND btrim(s3_bucket) <> '')
  )
);
