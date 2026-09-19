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

-- Reference/catalog data. Operational values remain in the existing telemetry
-- tables and are never seeded by this migration.

create table if not exists public.fleet_routes (
  route_id text primary key,
  name text not null,
  origin text,
  destination text,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.fleet_vehicles (
  bus_id text primary key,
  route_id text references public.fleet_routes(route_id),
  display_name text,
  capacity integer check (capacity is null or capacity > 0),
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.route_conditions (
  id bigserial primary key,
  route_id text not null references public.fleet_routes(route_id),
  congestion_score numeric check (
    congestion_score is null or congestion_score between 0 and 100
  ),
  pothole_count integer check (pothole_count is null or pothole_count >= 0),
  waterlogging_count integer check (
    waterlogging_count is null or waterlogging_count >= 0
  ),
  source text not null,
  observed_at timestamptz not null default now()
);

create table if not exists public.fitness_routes (
  id bigserial primary key,
  route_code text unique not null,
  name text not null,
  activity_type text not null check (
    activity_type in ('walking', 'jogging', 'cycling')
  ),
  distance_km numeric check (distance_km is null or distance_km > 0),
  duration_minutes integer check (
    duration_minutes is null or duration_minutes > 0
  ),
  latitude double precision,
  longitude double precision,
  geometry jsonb,
  verified boolean not null default false,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.sports_facilities (
  id bigserial primary key,
  facility_code text unique not null,
  name text not null,
  facility_type text not null,
  activities text[] not null default '{}',
  latitude double precision,
  longitude double precision,
  verified boolean not null default false,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.traffic_realtime
  add column if not exists route_id text references public.fleet_routes(route_id),
  add column if not exists bus_id text;

alter table public.road_damage
  add column if not exists route_id text references public.fleet_routes(route_id);

alter table public.vehicle_density
  add column if not exists route_id text references public.fleet_routes(route_id);

alter table public.incidents
  add column if not exists route_id text references public.fleet_routes(route_id),
  add column if not exists bus_id text;

alter table public.alerts
  add column if not exists route_id text references public.fleet_routes(route_id),
  add column if not exists bus_id text,
  add column if not exists title text,
  add column if not exists recommendation text;

create index if not exists route_conditions_route_observed_idx
  on public.route_conditions (route_id, observed_at desc);
create index if not exists traffic_realtime_route_recorded_idx
  on public.traffic_realtime (route_id, recorded_at desc);
create index if not exists road_damage_route_created_idx
  on public.road_damage (route_id, created_at desc);
create index if not exists vehicle_density_route_recorded_idx
  on public.vehicle_density (route_id, recorded_at desc);
create index if not exists incidents_route_timestamp_idx
  on public.incidents (route_id, timestamp desc);

-- Declared for fresh projects; existing installations already have this table.
create table if not exists public.buses (
  id bigserial primary key,
  -- One row per vehicle: telemetry upserts the latest position on this key.
  bus_id text not null unique,
  route_id text,
  latitude double precision,
  longitude double precision,
  speed numeric,
  heading numeric,
  status text,
  recorded_at timestamptz not null default now()
);

create index if not exists buses_bus_recorded_idx
  on public.buses (bus_id, recorded_at desc);

alter table public.fleet_routes enable row level security;
alter table public.fleet_vehicles enable row level security;
alter table public.route_conditions enable row level security;
alter table public.fitness_routes enable row level security;
alter table public.sports_facilities enable row level security;
alter table public.buses enable row level security;
alter table public.vehicle_density enable row level security;
alter table public.traffic_realtime enable row level security;
alter table public.ai_incidents enable row level security;
alter table public.road_damage_events enable row level security;

-- Seed reference metadata only. No telemetry, passenger counts, statuses,
-- predictions, or condition scores are inserted.
insert into public.fleet_routes (route_id, name, origin, destination)
values
  ('R001', 'Dilsukhnagar → Mehdipatnam', 'Dilsukhnagar', 'Mehdipatnam'),
  ('R002', 'Mehdipatnam → Dilsukhnagar', 'Mehdipatnam', 'Dilsukhnagar'),
  ('R003', 'LB Nagar → Secunderabad', 'LB Nagar', 'Secunderabad'),
  ('R004', 'Secunderabad → LB Nagar', 'Secunderabad', 'LB Nagar'),
  ('R005', 'Kukatpally → Ameerpet', 'Kukatpally', 'Ameerpet'),
  ('R006', 'Ameerpet → Kukatpally', 'Ameerpet', 'Kukatpally'),
  ('R007', 'Gachibowli → Secunderabad', 'Gachibowli', 'Secunderabad'),
  ('R008', 'Secunderabad → Gachibowli', 'Secunderabad', 'Gachibowli'),
  ('R009', 'Miyapur → Ameerpet', 'Miyapur', 'Ameerpet'),
  ('R010', 'Ameerpet → Miyapur', 'Ameerpet', 'Miyapur'),
  ('R011', 'Uppal → Mehdipatnam', 'Uppal', 'Mehdipatnam'),
  ('R012', 'Mehdipatnam → Uppal', 'Mehdipatnam', 'Uppal'),
  ('R013', 'Kondapur → Dilsukhnagar', 'Kondapur', 'Dilsukhnagar'),
  ('R014', 'Dilsukhnagar → Kondapur', 'Dilsukhnagar', 'Kondapur'),
  ('R015', 'Hitech City → Secunderabad', 'Hitech City', 'Secunderabad'),
  ('R016', 'Secunderabad → Hitech City', 'Secunderabad', 'Hitech City'),
  ('R017', 'LB Nagar → Mehdipatnam', 'LB Nagar', 'Mehdipatnam'),
  ('R018', 'Mehdipatnam → LB Nagar', 'Mehdipatnam', 'LB Nagar'),
  ('R019', 'Kukatpally → Gachibowli', 'Kukatpally', 'Gachibowli'),
  ('R020', 'Gachibowli → Kukatpally', 'Gachibowli', 'Kukatpally'),
  ('R021', 'Uppal → Secunderabad', 'Uppal', 'Secunderabad'),
  ('R022', 'Secunderabad → Uppal', 'Secunderabad', 'Uppal'),
  ('R023', 'Miyapur → Gachibowli', 'Miyapur', 'Gachibowli'),
  ('R024', 'Gachibowli → Miyapur', 'Gachibowli', 'Miyapur'),
  ('R025', 'Dilsukhnagar → Hitech City', 'Dilsukhnagar', 'Hitech City')
on conflict (route_id) do update set
  name = excluded.name,
  origin = excluded.origin,
  destination = excluded.destination,
  active = true,
  updated_at = now();

insert into public.fleet_vehicles (bus_id, route_id, display_name)
select
  'HYD-BUS-' || lpad(route_number::text, 3, '0'),
  'R' || lpad(route_number::text, 3, '0'),
  'Hyderabad Bus ' || lpad(route_number::text, 3, '0')
from generate_series(1, 25) as series(route_number)
on conflict (bus_id) do update set
  route_id = excluded.route_id,
  display_name = excluded.display_name,
  active = true,
  updated_at = now();

insert into public.fleet_vehicles (bus_id, route_id, display_name)
values
  ('TSRTC-001', 'R001', 'TSRTC-001'),
  ('TSRTC-002', 'R002', 'TSRTC-002'),
  ('TSRTC-003', 'R003', 'TSRTC-003')
on conflict (bus_id) do update set
  route_id = excluded.route_id,
  display_name = excluded.display_name,
  active = true,
  updated_at = now();
