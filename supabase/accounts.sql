-- Accounts row updated by POST /v1/webhooks/stripe.
-- Apply with the Supabase SQL editor or the Supabase CLI.
-- The API uses the service-role key, which bypasses row level security.
-- Do not put that key in the browser or in a committed env file.

create table if not exists public.accounts (
    id uuid primary key default gen_random_uuid(),
    client_domain text not null unique,
    brand_name text,
    stripe_customer_id text unique,
    stripe_subscription_id text,
    stripe_subscription_item_id text,
    subscription_status text,
    updated_at timestamptz not null default now()
);

create index if not exists accounts_subscription_item_idx
    on public.accounts (stripe_subscription_item_id);
