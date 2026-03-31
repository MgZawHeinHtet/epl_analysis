import pandas as pd
import requests
import io
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def scrape_player_stats(season="2024-25"):
    """
    Fetches player statistics from the FPL GitHub repository.
    Returns a pandas DataFrame with relevant columns.
    """
    url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        
        # Select relevant columns
        cols = [
            'web_name', 'team', 'element_type', 
            'goals_scored', 'assists', 'clean_sheets', 
            'goals_conceded', 'own_goals', 'penalties_saved', 
            'penalties_missed', 'yellow_cards', 'red_cards', 
            'saves', 'bonus', 'bps', 'influence', 'creativity', 
            'threat', 'ict_index', 'total_points', 'now_cost',
            'minutes', 'form'
        ]
        
        # Filter for existing columns only (in case CSV structure changes)
        existing_cols = [c for c in cols if c in df.columns]
        df = df[existing_cols]
        

        # Rename for better display
        rename_map = {
            'web_name': 'Player Name',
            'team': 'Team ID',
            'element_type': 'Position ID',
            'goals_scored': 'Goals',
            'assists': 'Assists',
            'clean_sheets': 'Clean Sheets',
            'total_points': 'Total Points',
            'now_cost': 'Price',
            'minutes': 'Minutes',
            'form': 'Form'
        }
        df = df.rename(columns=rename_map)
        
        # Map Team IDs to Team Names
        team_mapping = {
            1: 'Arsenal', 2: 'Aston Villa', 3: 'Bournemouth', 4: 'Brentford',
            5: 'Brighton', 6: 'Chelsea', 7: 'Crystal Palace', 8: 'Everton',
            9: 'Fulham', 10: 'Ipswich Town', 11: 'Leicester', 12: 'Liverpool',
            13: 'Man City', 14: 'Man United', 15: 'Newcastle', 16: "Nott'm Forest",
            17: 'Southampton', 18: 'Tottenham', 19: 'West Ham', 20: 'Wolves'
        }
        df['Team'] = df['Team ID'].map(team_mapping)
        
        # Convert Price (e.g., 100 -> 10.0)
        if 'Price' in df.columns:
            df['Price'] = df['Price'] / 10.0

        # Convert Stats to Numeric
        numeric_cols = ['influence', 'creativity', 'threat', 'ict_index']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                
        # Rename newly converted columns for display map
        rename_map_2 = {
            'influence': 'Influence',
            'creativity': 'Creativity',
            'threat': 'Threat',
            'ict_index': 'ICT Index'
        }
        df = df.rename(columns=rename_map_2)
            
        # Sort by Total Points by default
        df = df.sort_values('Total Points', ascending=False)
        
        return df

    except Exception as e:
        print(f"Error scraping player stats: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = scrape_player_stats()
    print(df.head())
