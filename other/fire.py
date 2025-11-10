import argparse
import json
import os
import sys
import requests
from pydantic import BaseModel
from google import genai
from tqdm import tqdm

# ------------------------------
# CONFIGURATION
# ------------------------------
BASE_URL = (
    "https://api.weather.gov/stations/{station}/observations/latest?require_qc=false"
)
HEADERS = {"accept": "application/geo+json"}
MODEL = "gemini-2.5-flash"


# ------------------------------
# STRUCTURED OUTPUT MODEL
# ------------------------------
class FireRisk(BaseModel):
    station: str
    likelihood: int  # 0–1000 scale


class FireRiskList(BaseModel):
    """List of fire risk assessments for multiple stations."""
    risks: list[FireRisk]


# ------------------------------
# FUNCTIONS
# ------------------------------
def fetch_weather_stations(limit: int | None = None) -> list[str]:
    """
    Fetch the list of weather observation stations from the NOAA API.
    
    Args:
        limit: Optional limit on the number of stations to fetch
        
    Returns:
        List of station URLs as strings
    """
    print("Fetching weather station list from API...")
    stations_url = "https://api.weather.gov/stations"
    if limit is not None:
        stations_url = f"{stations_url}?limit={limit}"
        print(f"  Requesting up to {limit} stations")
    
    stations_headers = {"accept": "application/geo+json"}
    response = requests.get(stations_url, headers=stations_headers)
    response.raise_for_status()
    
    data = response.json()
    stations = data['observationStations']
    
    print(f"  Found {len(stations)} weather stations")
    return stations


def extract_station_id(station_url: str) -> str:
    """
    Extract station ID from a station URL.
    
    Example: "https://api.weather.gov/stations/KBOS/observations" -> "KBOS"
    """
    # Station URLs are like: "https://api.weather.gov/stations/KBOS/observations"
    # Extract the station ID (the part after /stations/ and before /observations)
    parts = station_url.rstrip('/').split('/')
    # Find the index of 'stations' and return the next part
    try:
        stations_idx = parts.index('stations')
        return parts[stations_idx + 1]
    except (ValueError, IndexError):
        raise ValueError(f"Could not extract station ID from URL: {station_url}")


def fetch_weather_data(station_id: str) -> dict:
    """Fetch weather data for a given NOAA station."""
    url = BASE_URL.format(station=station_id)
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def analyze_fire_likelihoods(station_data: dict[str, dict]) -> list[FireRisk]:
    """
    Send weather data for multiple stations to Gemini and receive structured fire risk likelihoods.
    
    Args:
        station_data: Dictionary mapping station IDs to their weather data
        
    Returns:
        List of FireRisk assessments for each station
    """
    api_key = os.getenv("MM_GEMINI_API_KEY")
    if not api_key:
        raise ValueError("MM_GEMINI_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # Build prompt with all station data
    prompt_parts = [
        "You are a wildfire risk analyst. Given the following weather observation data for multiple stations, "
        "estimate the current fire likelihood for each station on a scale from 0 (no fire risk) to 1000 "
        "(extremely high fire risk). Return a list of results, each with two fields: "
        "`station` (the station ID) and `likelihood` (the integer fire likelihood).\n\n"
    ]
    
    for station_id, weather_data in station_data.items():
        prompt_parts.append(f"Station ID: {station_id}\n")
        prompt_parts.append(f"Weather data:\n{json.dumps(weather_data, indent=2)}\n\n")
    
    prompt = "".join(prompt_parts)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": FireRiskList,
        },
    )

    return response.parsed.risks


# ------------------------------
# MAIN SCRIPT
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze fire risk for weather stations using Gemini"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of stations to analyze (default: no limit)"
    )
    args = parser.parse_args()

    # Fetch the list of stations
    station_urls = fetch_weather_stations(limit=args.limit)
    
    if not station_urls:
        print("No stations found. Exiting.")
        sys.exit(1)
    
    # Extract station IDs from URLs and fetch weather data for each
    print(f"\nFetching weather data for {len(station_urls)} stations...")
    station_data = {}
    
    for station_url in tqdm(station_urls, desc="Fetching observations"):
        station_id = extract_station_id(station_url)
        try:
            weather_data = fetch_weather_data(station_id)
            station_data[station_id] = weather_data
        except Exception as e:
            print(f"\nWarning: Failed to fetch data for station {station_id}: {e}")
            continue
    
    if not station_data:
        print("No weather data retrieved. Exiting.")
        sys.exit(1)
    
    # Analyze all stations at once using structured outputs
    print(f"\nAnalyzing fire likelihood for {len(station_data)} stations using Gemini...")
    fire_risks = analyze_fire_likelihoods(station_data)

    # Print results as a list of FireRisk objects
    print("\nStructured Output:")
    output = [{"station": risk.station, "likelihood": risk.likelihood} for risk in fire_risks]
    print(json.dumps(output, indent=2))