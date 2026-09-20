import io
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from werkzeug.datastructures import FileStorage

from src.backend import app as app_module


class DataContractTests(unittest.TestCase):
    def test_model_weights_share_the_canonical_directory(self):
        expected = {
            "vehicle.pt",
            "garbage.pt",
            "pothole2v.pt",
            "license_plate.pt",
        }
        weights = app_module.PROJECT_ROOT / "weights"
        self.assertTrue(expected.issubset({path.name for path in weights.iterdir()}))

    def test_demand_excludes_operator_and_unconfigured_frames(self):
        now = datetime.now(timezone.utc)
        rows = [
            {
                "bus_id": "HYD-BUS-001",
                "person_count": 4,
                "vehicle_count": 1,
                "recorded_at": now.isoformat(),
                "source": "vehicle_detect",
            },
            {
                "bus_id": "OPERATOR-UPLOAD",
                "person_count": 99,
                "vehicle_count": 0,
                "recorded_at": now.isoformat(),
                "source": "vehicle_detect",
            },
            {
                "bus_id": "UNKNOWN-BUS",
                "person_count": 99,
                "vehicle_count": 0,
                "recorded_at": now.isoformat(),
                "source": "vehicle_detect",
            },
        ]

        def supabase_request(_method, table, **_kwargs):
            if table == "vehicle_density":
                return rows
            if table == "buses":
                return [{"bus_id": "HYD-BUS-001", "route_id": "R001"}]
            raise AssertionError(f"Unexpected table: {table}")

        with (
            patch.object(app_module, "supabase_request", side_effect=supabase_request),
            patch.object(
                app_module,
                "optional_supabase_rows",
                return_value=[{"bus_id": "HYD-BUS-001", "route_id": "R001"}],
            ),
        ):
            result = app_module.fetch_demand_forecast()

        self.assertEqual(result["data_quality"]["raw_frames"], 1)
        self.assertEqual(result["data_quality"]["excluded_frames"], 2)
        self.assertEqual(result["data_quality"]["buses_observed"], 1)
        self.assertEqual(result["routes"][0]["bus_id"], "HYD-BUS-001")

    def test_routes_score_two_fresh_signals_without_waterlogging(self):
        observed_at = datetime.now(timezone.utc).isoformat()
        routes = [
            {"route_id": "R001", "name": "Route One", "active": True},
            {"route_id": "R002", "name": "Route Two", "active": True},
        ]
        conditions = [
            {
                "route_id": "R001",
                "congestion_score": 20,
                "pothole_count": None,
                "waterlogging_count": None,
                "observed_at": observed_at,
            },
            {
                "route_id": "R001",
                "congestion_score": None,
                "pothole_count": 1,
                "waterlogging_count": None,
                "observed_at": observed_at,
            },
            {
                "route_id": "R002",
                "congestion_score": 40,
                "pothole_count": None,
                "waterlogging_count": None,
                "observed_at": observed_at,
            },
            {
                "route_id": "R002",
                "congestion_score": None,
                "pothole_count": 2,
                "waterlogging_count": None,
                "observed_at": observed_at,
            },
        ]

        def optional_rows(table, **_kwargs):
            return routes if table == "fleet_routes" else conditions

        with patch.object(
            app_module,
            "optional_supabase_rows",
            side_effect=optional_rows,
        ):
            result = app_module.fetch_routes()

        self.assertEqual(result["status"], "live")
        self.assertEqual(result["recommended"]["route_id"], "R001")
        self.assertEqual(
            result["recommended"]["fresh_signals"],
            ["congestion", "pothole"],
        )
        self.assertIsNone(result["recommended"]["waterlogging_count"])

    def test_facility_catalog_without_routes_is_not_live(self):
        def optional_rows(table, **_kwargs):
            if table == "fitness_routes":
                return []
            return [
                {
                    "facility_code": "HYD-SP-001",
                    "name": "Stadium",
                    "facility_type": "Arena",
                    "activities": None,
                    "verified": True,
                    "active": True,
                }
            ]

        with (
            patch.object(
                app_module,
                "optional_supabase_rows",
                side_effect=optional_rows,
            ),
            patch.object(app_module, "latest_traffic_snapshot", return_value=None),
        ):
            result = app_module.fetch_fitness()

        self.assertEqual(result["status"], "catalog_only")
        self.assertEqual(result["routes"], [])
        self.assertEqual(result["facilities"][0]["activities"], [])

    def test_stale_bus_does_not_claim_active_operational_status(self):
        old_stamp = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()

        def supabase_request(_method, table, **_kwargs):
            self.assertEqual(table, "buses")
            return [
                {
                    "bus_id": "HYD-BUS-001",
                    "route_id": "R001",
                    "status": "ACTIVE",
                    "recorded_at": old_stamp,
                }
            ]

        def optional_rows(table, **_kwargs):
            if table == "fleet_vehicles":
                return [{"bus_id": "HYD-BUS-001", "route_id": "R001"}]
            return [{"route_id": "R001", "name": "Route One"}]

        with (
            patch.object(app_module, "supabase_request", side_effect=supabase_request),
            patch.object(
                app_module,
                "optional_supabase_rows",
                side_effect=optional_rows,
            ),
        ):
            result = app_module.fetch_fleet()

        bus = result["buses"][0]
        self.assertEqual(bus["telemetry_status"], "stale")
        self.assertEqual(bus["operational_status"], "UNKNOWN")
        self.assertEqual(bus["last_reported_operational_status"], "ACTIVE")

    def test_upload_validation_and_cleanup(self):
        app = app_module.app
        with app.test_request_context("/api/vehicle-detect", method="POST"):
            upload = FileStorage(
                stream=io.BytesIO(b"fake-image"),
                filename="frame.jpg",
                content_type="image/jpeg",
            )
            path = app_module.save_upload(upload, "test")
            self.assertTrue(path.exists())
            app_module.cleanup_uploads(None)
            self.assertFalse(path.exists())

        client = app.test_client()
        response = client.post(
            "/api/vehicle-detect",
            data={
                "image": (
                    io.BytesIO(b"not-an-image"),
                    "payload.txt",
                    "text/plain",
                )
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only JPG", response.get_json()["error"])

    def test_telemetry_rejects_coordinates_outside_hyderabad(self):
        with self.assertRaisesRegex(ValueError, "latitude must be between"):
            app_module.build_telemetry_row(
                {
                    "bus_id": "HYD-BUS-001",
                    "latitude": 0,
                    "longitude": 20,
                },
                {"HYD-BUS-001": "R001"},
                datetime.now(timezone.utc).isoformat(),
            )

    def test_health_is_degraded_when_a_core_model_cannot_load(self):
        with patch.object(
            app_module,
            "model_readiness",
            return_value=(
                {
                    "vehicle": True,
                    "garbage": True,
                    "pothole": False,
                    "plate": True,
                },
                {"pothole": "load failed"},
            ),
        ):
            response = app_module.app.test_client().get("/api/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["status"], "degraded")
        self.assertEqual(response.get_json()["failed_models"], ["pothole"])

    def test_cors_does_not_allow_unknown_origins(self):
        response = app_module.app.test_client().options(
            "/api/health",
            headers={
                "Origin": "https://example.invalid",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    def test_cors_wildcard_configuration_is_ignored(self):
        origins = app_module.build_cors_origins(
            "*,https://preview.example.com"
        )
        self.assertNotIn("*", origins)
        self.assertIn("https://preview.example.com", origins)
        self.assertIn(
            "https://hyderabad-urban-intelligence.vercel.app",
            origins,
        )


if __name__ == "__main__":
    unittest.main()
