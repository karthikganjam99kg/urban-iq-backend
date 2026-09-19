-- UrbanIQ: move urban_data.json into Postgres so alerts/incidents survive
-- container restarts and no longer ship inside the repo.
--
-- Run once in Supabase Studio -> SQL Editor.
-- Reads/writes go through the Flask API with the secret key, so RLS stays
-- closed to anonymous clients.

-- 1. Alerts had no table at all.
create table if not exists public.alerts (
  id bigserial primary key,
  alert_id text unique not null,
  incident_id text,
  alert_type text,
  severity text,
  location text,
  message text,
  status text not null default 'New',
  detected_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists alerts_detected_at_idx
  on public.alerts (detected_at desc);

-- 2. The existing incidents table lacks the fields the JSON carries.
alter table public.incidents
  add column if not exists severity text,
  add column if not exists location text,
  add column if not exists description text,
  add column if not exists status text;

-- incident_id is the natural key coming from the JSON payloads.
create unique index if not exists incidents_incident_id_key
  on public.incidents (incident_id);

-- 3. Keep anonymous browser keys out of both tables.
alter table public.alerts enable row level security;
alter table public.incidents enable row level security;
