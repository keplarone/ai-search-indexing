-- One row per keyword on each completed audit.
-- Apply with the Supabase SQL editor. The API uses the service-role key.

create table if not exists public.audit_transactions (
    id uuid primary key default gen_random_uuid(),
    client_domain text not null,
    brand_name text not null,
    keyword_queried text not null,
    vibe_share_score double precision not null,
    competitors_detected jsonb not null default '[]'::jsonb,
    openai_status text not null,
    perplexity_status text not null,
    created_at timestamptz not null default now()
);

create index if not exists audit_transactions_domain_idx
    on public.audit_transactions (client_domain);
