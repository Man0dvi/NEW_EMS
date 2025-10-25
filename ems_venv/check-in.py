CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

CREATE TABLE IF NOT EXISTS deep_research_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID,
    source_url TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    text_content TEXT NOT NULL,
    keywords TEXT[], -- PostgreSQL Array type for lists of strings
    embedding_vector VECTOR(1536) -- Define the vector dimension (e.g., 1536 for text-embedding-3-large)
);

-- Optional: Create GIN index for faster keyword array searches
CREATE INDEX idx_keywords ON deep_research_chunks USING GIN (keywords);

-- Optional: Create an index on document_id for faster grouping by source (Agent 3)
CREATE INDEX idx_document_id ON deep_research_chunks (document_id);
