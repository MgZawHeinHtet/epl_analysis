import requests
import pandas as pd
import io

# Try the GitHub raw CSV URL for 2024-25 season (current season)
url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv"

print(f"Fetching data from {url}...")
try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    # Read CSV from response content
    df = pd.read_csv(io.StringIO(response.text))
    
    print(f"Successfully fetched {len(df)} players.")
    print(df[['web_name', 'goals_scored', 'assists', 'total_points']].head())
    
    # Save for inspection
    df.to_csv('player_stats.csv', index=False)

except Exception as e:
    print(f"Error: {e}")
