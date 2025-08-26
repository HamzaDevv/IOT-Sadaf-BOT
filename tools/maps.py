# tools/maps_tool.py
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# Load API key from environment variable
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

IP_GEO_API = "http://ip-api.com/json/"  # free IP-based location service


def get_current_location() -> str:
    """Get current location based on IP address."""
    try:
        resp = requests.get(IP_GEO_API, timeout=5).json()
        if resp["status"] == "success":
            city = resp.get("city", "unknown")
            region = resp.get("regionName", "unknown")
            country = resp.get("country", "unknown")
            lat = resp.get("lat", "unknown")
            lon = resp.get("lon", "unknown")
            org = resp.get("org", "unknown")
            return f"📍 You are in city : {city}, region : {region}, country : {country} , organization : {org} (Latitue: {lat}, Longitude: {lon})"
        else:
            return "❌ Unable to detect current location."
    except Exception as e:
        return f"Error detecting location: {e}"


def search_places(query: str, location: str = None, radius: int = 5000) -> str:
    """Search for places (e.g., restaurants, gyms)."""
    if not GOOGLE_MAPS_API_KEY:
        return "❌ Google Maps API key is missing."

    try:
        params = {
            "query": query,
            "key": GOOGLE_MAPS_API_KEY,
        }
        if location:
            params["location"] = location
            params["radius"] = radius

        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        resp = requests.get(url, params=params).json()

        if not resp.get("results"):
            return f"No results found for '{query}'."

        results = []
        for place in resp["results"][:5]:  # top 5 results
            name = place.get("name", "Unknown")
            addr = place.get("formatted_address", "No address available")
            rating = place.get("rating", "N/A")
            results.append(f"🏷 {name}\n   📍 {addr}\n   ⭐ {rating}\n")
        return "\n\n".join(results)

    except Exception as e:
        return f"Error searching places: {e}"


def get_place_details(place_id: str) -> str:
    """Get detailed info about a place using place_id."""
    if not GOOGLE_MAPS_API_KEY:
        return "❌ Google Maps API key is missing."

    try:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {"place_id": place_id, "key": GOOGLE_MAPS_API_KEY}
        resp = requests.get(url, params=params).json()

        if "result" not in resp:
            return "No details found."

        result = resp["result"]
        name = result.get("name", "Unknown")
        addr = result.get("formatted_address", "No address")
        phone = result.get("formatted_phone_number", "No phone")
        rating = result.get("rating", "N/A")
        hours = (
            ", ".join(result.get("weekday_text", []))
            if "weekday_text" in result
            else "No timings"
        )
        website = result.get("website", "No website")

        return (
            f"🏷 {name}\n"
            f"📍 Address: {addr}\n"
            f"📞 Phone: {phone}\n"
            f"⭐ Rating: {rating}\n"
            f"⏰ Hours: {hours}\n"
            f"🔗 Website: {website}"
        )

    except Exception as e:
        return f"Error fetching place details: {e}"


def nearby_search(location: str, place_type: str, radius: int = 2000) -> str:
    """Find nearby places of a certain type (e.g., pharmacy, ATM)."""
    if not GOOGLE_MAPS_API_KEY:
        return "❌ Google Maps API key is missing."

    try:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": location,
            "radius": radius,
            "type": place_type,
            "key": GOOGLE_MAPS_API_KEY,
        }
        resp = requests.get(url, params=params).json()

        if not resp.get("results"):
            return f"No nearby {place_type} found."

        results = []
        for place in resp["results"][:5]:  # top 5
            name = place.get("name", "Unknown")
            vicinity = place.get("vicinity", "No address")
            rating = place.get("rating", "N/A")
            results.append(f"🏷 {name}\n   📍 {vicinity}\n   ⭐ {rating}")
        return "\n\n".join(results)

    except Exception as e:
        return f"Error in nearby search: {e}"


def route_planning(origin: str, destination: str, mode: str = "driving") -> str:
    """Get step-by-step directions from A → B."""
    if not GOOGLE_MAPS_API_KEY:
        return "❌ Google Maps API key is missing."

    try:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "key": GOOGLE_MAPS_API_KEY,
        }
        resp = requests.get(url, params=params).json()

        if not resp.get("routes"):
            return f"No route found from {origin} to {destination}."

        steps = []
        for step in resp["routes"][0]["legs"][0]["steps"]:
            instruction = step["html_instructions"]
            distance = step["distance"]["text"]
            duration = step["duration"]["text"]

            # Remove HTML tags from instructions
            import re

            instruction_clean = re.sub("<.*?>", "", instruction)

            steps.append(f"➡ {instruction_clean} ({distance}, {duration})")

        return "\n".join(steps)

    except Exception as e:
        return f"Error in route planning: {e}"


if __name__ == "__main__":
    # Example usage
    # print(get_current_location())
    print("\n--- Search Places ---")
    print(search_places("B.E College"))
    # print("\n--- Nearby Search ---")
    # print(nearby_search("22.5643,88.3693", "atm", 1000))
    # print("\n--- Route Planning ---")
    # print(route_planning("Howrah station", "BE college shibpur west bengal"))
