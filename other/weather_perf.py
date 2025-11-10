import argparse
import requests
import time
from tqdm import tqdm

# run this file with python weather_perf.py --limit 100


# API endpoint constants for weather station data
weather_url = "/latest?require_qc=false"
weather_headers = {"accept": "application/geo+json"}


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
    response.raise_for_status()  # Raise exception for bad status codes
    
    data = response.json()
    stations = data['observationStations']
    
    print(f"  Found {len(stations)} weather stations")
    return stations


def main(limit: int | None = None) -> None:
    """
    Main function that benchmarks weather API request times.
    
    Fetches a list of weather stations, then makes requests to each station's
    latest observation endpoint to measure response times.
    """
    # Track unique HTTP headers seen across all responses
    unique_headers = {}
    # Store timing data for each request
    timings = []

    # Fetch the list of observation stations to test
    stations = fetch_weather_stations(limit)
    
    if not stations:
        print("No stations found. Exiting.")
        return
    
    print(f"\nBenchmarking {len(stations)} station requests...")
    
    # Time the requests to the weather API for each station
    # tqdm provides a progress bar to show request progress
    for station in tqdm(stations, desc="Processing stations"):
        # Measure the time it takes to get the latest observations from this station
        start = time.perf_counter()
        response = requests.get(station + weather_url, headers=weather_headers)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

        # Track unique headers and values observed across all responses
        # This helps identify caching, rate limiting, or other header-based behavior
        for key, val in response.headers.items():
            if key not in unique_headers:
                unique_headers[key] = set()
            unique_headers[key].add(val)

    # Print performance statistics
    print(f"\nResults:")
    print(f"  Average request time: {sum(timings) / len(timings):.4f} seconds")
    print(f"  Min time: {min(timings):.4f} seconds, Max time: {max(timings):.4f} seconds")
    # print("\nUnique headers and values observed:")
    # for k, v in unique_headers.items():
    #     print(f"{k}: {sorted(v)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark weather API request times"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of stations to fetch (default: no limit)"
    )
    args = parser.parse_args()
    main(limit=args.limit)


