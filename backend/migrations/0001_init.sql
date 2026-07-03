CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  filename     text NOT NULL,
  storage_path text NOT NULL,
  mime_type    text,
  status       text NOT NULL DEFAULT 'processing',  -- processing | ready | failed
  summary      text,
  tags         text[],
  error        text,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id          bigserial PRIMARY KEY,
  document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index int NOT NULL,
  content     text NOT NULL,
  embedding   vector(1024)
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
