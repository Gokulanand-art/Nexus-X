-- supabase_schema.sql — one-time setup for the Nexus v2 cloud RAG backend.
-- Run this in the Supabase SQL editor (https://supabase.com/dashboard →
-- your project → SQL Editor → New query → paste → Run).
--
-- This creates: pgvector extension, the nexus_chunks table (768-dim
-- embeddings to match nomic-embed-text), a generated tsvector column for
-- full-text search, GIN indexes, and the hybrid_search() function that
-- fuses semantic + keyword results (RRF-style).

create extension if not exists vector;

create table if not exists nexus_chunks (
    chunk_id  text primary key,
    text      text not null,
    source    text not null,
    metadata  jsonb not null default '{}'::jsonb,
    embedding vector(768) not null,
    created   timestamptz not null default now()
);

-- Full-text keyword side
alter table nexus_chunks add column if not exists search_text tsvector
    generated always as (
        to_tsvector('english', coalesce(text, ''))
    ) stored;

create index if not exists nexus_chunks_embedding_idx
    on nexus_chunks using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create index if not exists nexus_chunks_fts_idx
    on nexus_chunks using gin (search_text);

create index if not exists nexus_chunks_source_idx on nexus_chunks (source);

-- Hybrid search: RRF fusion of vector cosine ranks + ts_rank keyword ranks
create or replace function hybrid_search(
    query_text text,
    query_vec  text,
    top_k      int default 8,
    source_filter text default ''
)
returns table (chunk_id text, text_out text, source text, metadata jsonb, score float)
language plpgsql
as $$
declare
    vec vector(768) := query_vec::vector;
begin
    return query
    with vector_hits as (
        select chunk_id, 1.0 / (60 + row_number() over (order by embedding <=> vec)) as vscore
        from nexus_chunks
        order by embedding <=> vec
        limit top_k * 4
    ),
    keyword_hits as (
        select chunk_id, 1.0 / (60 + row_number() over (order by ts_rank(search_text, plainto_tsquery('english', query_text)) desc)) as kscore
        from nexus_chunks
        where search_text @@ plainto_tsquery('english', query_text)
        order by ts_rank(search_text, plainto_tsquery('english', query_text)) desc
        limit top_k * 4
    )
    select c.chunk_id,
           c.text as text_out,
           c.source,
           c.metadata,
           (coalesce(v.vscore, 0) + coalesce(k.kscore, 0))::float as score
    from nexus_chunks c
    left join vector_hits v  on v.chunk_id = c.chunk_id
    left join keyword_hits k on k.chunk_id = c.chunk_id
    where (v.chunk_id is not null or k.chunk_id is not null)
      and (source_filter = '' or c.source like '%' || source_filter || '%')
    order by score desc
    limit top_k;
end;
$$;

-- Helpers used by /memory, /clear
create or replace function nexus_stats()
returns table (total bigint)
language sql as $$
    select count(*) from nexus_chunks;
$$;

create or replace function nexus_clear()
returns void
language sql as $$
    delete from nexus_chunks;
$$;
