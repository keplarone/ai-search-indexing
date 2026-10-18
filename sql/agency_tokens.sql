-- Hashed agency API tokens. The plaintext token is never stored.
-- Apply with the Supabase SQL editor. The API uses the service-role key.

create table if not exists public.agency_tokens (
    id uuid primary key default gen_random_uuid(),
    token_hash text not null unique,
    allotted_credits integer not null check (allotted_credits > 0),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists agency_tokens_email_idx
    on public.agency_tokens ((metadata->>'client_email'));
