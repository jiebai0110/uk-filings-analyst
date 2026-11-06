CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    company     TEXT NOT NULL,
    year        INT  NOT NULL,
    page        INT  NOT NULL,
    chunk_index INT  NOT NULL,
    source_file TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(1536),
    -- full-text column, used for BM25-style lexical retrieval in week 2
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (source_file, page, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_company_year_idx ON chunks (company, year);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
