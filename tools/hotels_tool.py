import json

def search_hotels(destination: str):
    """
    Searches for hotels in a given destination.

    Args:
        destination: The destination to search for hotels in.

    Returns:
        A JSON string containing a list of hotels with their name, rating, and price.
    """
    print(f"Searching for hotels in {destination}...")

    mock_hotels = [
        {"name": "Grand Hyatt", "rating": 4.5, "price": "$250"},
        {"name": "Hilton Garden Inn", "rating": 4.2, "price": "$180"},
        {"name": "Marriott Marquis", "rating": 4.8, "price": "$350"}
    ]

    return json.dumps(mock_hotels)