-- Canonical UrbanIQ schema (urban-iq-backend).
-- Run in Supabase Studio -> SQL Editor. Safe to re-run (IF NOT EXISTS).
--
-- Existing operational tables (do not recreate as traffic_snapshots / detections):
--   traffic_realtime   TomTom flow (Flask writes with the secret key)
--   road_damage        detections the browser may insert (anon RLS open)
--   road_damage_events hazard events (anon insert blocked)
--   vehicle_density    per-frame counts (anon insert blocked)
--   ai_incidents       rash-motion / plate candidates (anon insert blocked)
--   incidents          civic incidents (Flask secret key)
--   buses              fleet positions
--   alerts             civic alerts (Flask secret key)

create index if not exists traffic_realtime_recorded_at_idx
  on public.traffic_realtime (recorded_at desc);

create index if not exists road_damage_created_at_idx
  on public.road_damage (created_at desc);

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

alter table public.incidents
  add column if not exists severity text,
  add column if not exists location text,
  add column if not exists description text,
  add column if not exists status text;

create unique index if not exists incidents_incident_id_key
  on public.incidents (incident_id);

alter table public.alerts enable row level security;
alter table public.incidents enable row level security;
