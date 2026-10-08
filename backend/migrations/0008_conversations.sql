-- Conversation history (issue #52). /api/chat stays stateless; the frontend
-- saves each finished message through /api/conversations.
--
-- There is no login yet, so the owner is an anonymous browser: owner_id is the
-- SHA-256 (hex) of its knowva_client cookie token, never the token itself. A
-- later login can hand a browser's conversations to an account with one UPDATE
-- of owner_kind and owner_id.

CREATE TABLE conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_kind text NOT NULL CHECK (owner_kind IN ('anonymous', 'user')),
  owner_id text NOT NULL CHECK (owner_id <> ''),
  title text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  -- An anonymous owner is only ever stored as a token hash.
  CONSTRAINT conversations_anonymous_owner_is_hash_check
    CHECK (owner_kind <> 'anonymous' OR owner_id ~ '^[0-9a-f]{64}$')
);

CREATE INDEX conversations_owner_recent_idx
  ON conversations (owner_kind, owner_id, updated_at DESC);

CREATE TABLE conversation_messages (
  conversation_id uuid NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
  message_id text NOT NULL CHECK (char_length(message_id) BETWEEN 1 AND 128),
  parent_id text CHECK (parent_id IS NULL OR char_length(parent_id) BETWEEN 1 AND 128),
  -- Save order within the conversation; a resent message keeps its place.
  position integer NOT NULL CHECK (position >= 0),
  -- The whole assistant-ui message, sources data part included.
  message jsonb NOT NULL CHECK (jsonb_typeof(message) = 'object'),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (conversation_id, message_id),
  UNIQUE (conversation_id, position)
);
