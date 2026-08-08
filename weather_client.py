"""
Weather API Client (weather_client.py)

Fetches active weather alerts and detailed narrative forecasts from the
National Weather Service API (https://api.weather.gov) and normalizes them
into a standardized document schema for vector ingestion and retrieval.
"""

import hashlib
import json
import logging
import re
import requests

logger = logging.getLogger("weather-client")

DEFAULT_USER_AGENT = "(databricks-bootcamp-weather-app, student@example.com)"
BASE_URL = "https://api.weather.gov"

# Popular city/state mappings to (lat, lon, state)
CITY_COORDINATES = {
    "CHICAGO, IL": (41.8781, -87.6298, "IL"),
    "CHICAGO": (41.8781, -87.6298, "IL"),
    "AUSTIN, TX": (30.2672, -97.7431, "TX"),
    "AUSTIN": (30.2672, -97.7431, "TX"),
    "NEW YORK, NY": (40.7128, -74.0060, "NY"),
    "NEW YORK": (40.7128, -74.0060, "NY"),
    "LOS ANGELES, CA": (34.0522, -118.2437, "CA"),
    "LOS ANGELES": (34.0522, -118.2437, "CA"),
    "SEATTLE, WA": (47.6062, -122.3321, "WA"),
    "SEATTLE": (47.6062, -122.3321, "WA"),
    "MIAMI, FL": (25.7617, -80.1918, "FL"),
    "MIAMI": (25.7617, -80.1918, "FL"),
    "DENVER, CO": (39.7392, -104.9903, "CO"),
    "DENVER": (39.7392, -104.9903, "CO"),
}


class WeatherClient:
    def __init__(self, user_agent: str = DEFAULT_USER_AGENT):
        self.headers = {"User-Agent": user_agent, "Accept": "application/geo+json"}

    def fetch_active_alerts(self, state: str, limit: int = 50) -> list[dict]:
        """
        Fetch active alerts for a specific US state code (e.g., 'IL', 'TX').
        """
        state = state.strip().upper()
        url = f"{BASE_URL}/alerts/active?area={state}"
        logger.info(f"Fetching active weather alerts for state: {state}")

        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            documents = []

            for feat in features[:limit]:
                props = feat.get("properties", {})
                alert_id = props.get("id") or feat.get("id")
                event = props.get("event") or props.get("headline") or "Weather Alert"
                desc = props.get("description") or ""
                instruction = props.get("instruction") or ""

                narrative_parts = []
                if desc.strip():
                    narrative_parts.append(desc.strip())
                if instruction.strip():
                    narrative_parts.append(f"INSTRUCTIONS: {instruction.strip()}")

                narrative_text = "\n\n".join(narrative_parts)
                if not narrative_text:
                    continue

                location = props.get("areaDesc") or state
                issued_at = props.get("sent") or props.get("effective") or props.get("onset")

                documents.append({
                    "id": alert_id,
                    "location": location,
                    "source_type": "alert",
                    "headline": event,
                    "narrative_text": narrative_text,
                    "issued_at": issued_at,
                    "payload": feat,
                })

            return documents

        except Exception as e:
            logger.error(f"Failed to fetch alerts for {state}: {e}")
            return []

    def fetch_forecast_discussion(self, lat: float, lon: float, location_name: str = None) -> list[dict]:
        """
        Fetch detailed narrative forecast periods for a given lat/lon location.
        """
        loc_str = location_name or f"{lat:.4f},{lon:.4f}"
        logger.info(f"Resolving forecast point for: {loc_str}")

        try:
            point_url = f"{BASE_URL}/points/{lat:.4f},{lon:.4f}"
            resp = requests.get(point_url, headers=self.headers, timeout=10)
            resp.raise_for_status()
            point_data = resp.json()

            forecast_url = point_data.get("properties", {}).get("forecast")
            if not forecast_url:
                logger.warning(f"No forecast URL found for point: {loc_str}")
                return []

            fc_resp = requests.get(forecast_url, headers=self.headers, timeout=10)
            fc_resp.raise_for_status()
            fc_data = fc_resp.json()

            periods = fc_data.get("properties", {}).get("periods", [])
            documents = []

            for p in periods:
                name = p.get("name", "Forecast Period")
                short_fc = p.get("shortForecast", "")
                detailed_fc = p.get("detailedForecast", "")

                if not detailed_fc.strip():
                    continue

                headline = f"{loc_str} Forecast ({name}): {short_fc}"
                issued_at = p.get("startTime")

                # Generate a stable dedup key for forecast periods
                hash_input = f"{loc_str}_{name}_{issued_at}".encode("utf-8")
                doc_id = f"fc_{hashlib.md5(hash_input).hexdigest()}"

                documents.append({
                    "id": doc_id,
                    "location": loc_str,
                    "source_type": "forecast",
                    "headline": headline,
                    "narrative_text": detailed_fc,
                    "issued_at": issued_at,
                    "payload": p,
                })

            return documents

        except Exception as e:
            logger.error(f"Failed to fetch forecast for {loc_str}: {e}")
            return []

    def harvest_locations(self, locations: list[str], limit_per_location: int = 50) -> list[dict]:
        """
        Harvest weather records (alerts + forecasts) for a list of location specifications.
        Locations can be state codes ('IL'), city strings ('Chicago, IL'), or 'lat,lon' strings.
        """
        all_docs = []
        seen_ids = set()

        for loc in locations:
            loc_clean = loc.strip().upper()
            if not loc_clean:
                continue

            # 1. State code (2 letters)
            if len(loc_clean) == 2 and loc_clean.isalpha():
                alerts = self.fetch_active_alerts(loc_clean, limit=limit_per_location)
                for d in alerts:
                    if d["id"] not in seen_ids:
                        seen_ids.add(d["id"])
                        all_docs.append(d)

            # 2. Known City lookup
            elif loc_clean in CITY_COORDINATES:
                lat, lon, state = CITY_COORDINATES[loc_clean]
                alerts = self.fetch_active_alerts(state, limit=limit_per_location)
                for d in alerts:
                    if d["id"] not in seen_ids:
                        seen_ids.add(d["id"])
                        all_docs.append(d)
                forecasts = self.fetch_forecast_discussion(lat, lon, location_name=loc)
                for d in forecasts:
                    if d["id"] not in seen_ids:
                        seen_ids.add(d["id"])
                        all_docs.append(d)

            # 3. Lat/Lon pair (e.g. "41.8781,-87.6298")
            elif "," in loc_clean:
                parts = loc_clean.split(",")
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    forecasts = self.fetch_forecast_discussion(lat, lon, location_name=loc)
                    for d in forecasts:
                        if d["id"] not in seen_ids:
                            seen_ids.add(d["id"])
                            all_docs.append(d)
                except ValueError:
                    logger.warning(f"Could not parse lat/lon pair from: {loc}")

            else:
                logger.warning(f"Unrecognized location specification: {loc}")

        return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = WeatherClient()
    docs = client.harvest_locations(["IL", "Chicago, IL", "Austin, TX"])
    print(f"Harvested {len(docs)} documents.")
    if docs:
        print("Sample Document:", json.dumps(docs[0], indent=2, default=str))
