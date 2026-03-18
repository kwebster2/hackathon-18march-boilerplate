"""
Integration tests against the real Tour Guide mock API.
No LLM required — verifies our API connectivity and data shapes.

Run: pytest tests/test_api.py -v
"""

import pytest
import httpx

BASE = "https://hacketon-18march-api.orcaplatform.ai/tour-guide-1/api"
HEADERS = {"X-API-Key": "tour-guide-1-key-vwx234"}


@pytest.fixture(scope="module")
def client():
    with httpx.Client(headers=HEADERS, timeout=10) as c:
        yield c


# ── Tour listing ──────────────────────────────────────────────────────────────

class TestTourListing:
    def test_list_all_tours(self, client):
        r = client.get(f"{BASE}/tours")
        assert r.status_code == 200
        tours = r.json()
        assert isinstance(tours, list)
        assert len(tours) > 0

    def test_tours_have_required_fields(self, client):
        tours = client.get(f"{BASE}/tours").json()
        for tour in tours:
            assert "id" in tour
            assert "name" in tour
            assert "category" in tour
            assert "difficulty" in tour

    def test_filter_by_category(self, client):
        r = client.get(f"{BASE}/tours", params={"category": "adventure"})
        assert r.status_code == 200
        tours = r.json()
        assert all(t["category"] == "adventure" for t in tours)

    def test_filter_by_difficulty(self, client):
        r = client.get(f"{BASE}/tours", params={"difficulty": "easy"})
        assert r.status_code == 200
        tours = r.json()
        assert all(t["difficulty"] == "easy" for t in tours)

    def test_filter_by_max_price(self, client):
        r = client.get(f"{BASE}/tours", params={"max_price": 50})
        assert r.status_code == 200
        tours = r.json()
        # All returned tours should have price <= 50
        for tour in tours:
            price = tour.get("price_per_person") or tour.get("price") or 0
            assert price <= 50

    def test_list_categories(self, client):
        r = client.get(f"{BASE}/categories")
        assert r.status_code == 200

    def test_invalid_api_key_returns_401(self):
        with httpx.Client(timeout=10) as c:
            r = c.get(f"{BASE}/tours", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401


# ── Single tour ───────────────────────────────────────────────────────────────

class TestTourDetails:
    @pytest.fixture(scope="class")
    def first_tour_id(self, client):
        tours = client.get(f"{BASE}/tours").json()
        return tours[0]["id"]

    def test_get_tour_by_id(self, client, first_tour_id):
        r = client.get(f"{BASE}/tours/{first_tour_id}")
        assert r.status_code == 200
        tour = r.json()
        assert tour["id"] == first_tour_id

    def test_get_nonexistent_tour_returns_404(self, client):
        r = client.get(f"{BASE}/tours/99999")
        assert r.status_code == 404


# ── Pricing ───────────────────────────────────────────────────────────────────

class TestPricing:
    @pytest.fixture(scope="class")
    def first_tour_id(self, client):
        return client.get(f"{BASE}/tours").json()[0]["id"]

    def test_get_pricing_single_guest(self, client, first_tour_id):
        r = client.get(f"{BASE}/pricing", params={"tour_id": first_tour_id, "guests": 1})
        assert r.status_code == 200
        data = r.json()
        assert "total" in data or "total_price" in data or "price" in data

    def test_pricing_scales_with_guests(self, client, first_tour_id):
        r1 = client.get(f"{BASE}/pricing", params={"tour_id": first_tour_id, "guests": 1}).json()
        r4 = client.get(f"{BASE}/pricing", params={"tour_id": first_tour_id, "guests": 4}).json()

        total_1 = r1.get("total") or r1.get("total_price") or r1.get("price", 0)
        total_4 = r4.get("total") or r4.get("total_price") or r4.get("price", 0)
        assert total_4 > total_1


# ── Availability ──────────────────────────────────────────────────────────────

class TestAvailability:
    @pytest.fixture(scope="class")
    def first_tour_id(self, client):
        return client.get(f"{BASE}/tours").json()[0]["id"]

    def test_check_availability_returns_200(self, client, first_tour_id):
        r = client.get(
            f"{BASE}/tours/available",
            params={"tour_id": first_tour_id, "date": "2026-04-15"}
        )
        assert r.status_code == 200

    def test_availability_includes_spots_field(self, client, first_tour_id):
        data = client.get(
            f"{BASE}/tours/available",
            params={"tour_id": first_tour_id, "date": "2026-04-15"}
        ).json()
        # Should have some field indicating remaining spots
        keys = set(data.keys())
        assert keys & {"available_spots", "remaining_spots", "spots", "available", "available_capacity"}


# ── Booking lifecycle ─────────────────────────────────────────────────────────

class TestBookingLifecycle:
    """Full create → read → cancel flow using a real tour ID."""

    @pytest.fixture(scope="class")
    def first_tour_id(self, client):
        return client.get(f"{BASE}/tours").json()[0]["id"]

    @pytest.fixture(scope="class")
    def booking(self, client, first_tour_id):
        payload = {
            "tour_id": first_tour_id,
            "tour_date": "2026-04-20",
            "guest_name": "Test User",
            "guest_email": "test@example.com",
            "num_guests": 2,
        }
        r = client.post(f"{BASE}/bookings", json=payload)
        assert r.status_code in (200, 201), f"Booking failed: {r.text}"
        return r.json()

    def test_booking_has_id(self, booking):
        assert "id" in booking

    def test_get_booking_by_id(self, client, booking):
        r = client.get(f"{BASE}/bookings/{booking['id']}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == booking["id"]

    def test_booking_appears_in_list(self, client, booking):
        bookings = client.get(f"{BASE}/bookings").json()
        ids = [b["id"] for b in bookings]
        assert booking["id"] in ids

    def test_cancel_booking(self, client, booking):
        r = client.delete(f"{BASE}/bookings/{booking['id']}")
        assert r.status_code in (200, 204)

    def test_cancelled_booking_has_cancelled_status(self, client, booking):
        r = client.get(f"{BASE}/bookings/{booking['id']}")
        if r.status_code == 200:
            assert r.json().get("status") == "cancelled"
