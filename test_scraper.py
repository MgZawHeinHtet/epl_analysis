import requests
import pandas as pd

url = "https://fantasy.premierleague.com/api/bootstrap-static/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print(f"Fetching data from {url}...")
try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    players = data['elements']
    df = pd.DataFrame(players)
    
    print(f"Successfully fetched {len(df)} players.")
    print(df[['web_name', 'goals_scored', 'assists', 'total_points']].head())
    
    # Save a small sample to inspect columns if needed
    df.head().to_csv('player_stats_sample.csv', index=False)

except Exception as e:
    print(f"Error: {e}")
