import streamlit as st
import pandas as pd
import numpy as np
from feature_engineer import load_and_prepare_data, create_team_features
from scraper import scrape_player_stats

st.set_page_config(layout="wide", page_title="Premier League Master")
st.title("🏆 PL Unified Predictor & Analysis")

# ဒေတာ Load လုပ်ခြင်း
df = load_and_prepare_data()

# testing for nice 

if df.empty:
    st.error("❌ ဒေတာဖတ်လို့မရပါ။ 'data/' folder ထဲက ဖိုင်တွေကို စစ်ဆေးပေးပါ။")
else:
    all_seasons = sorted(df['Season'].unique(), reverse=True)
    current_season = all_seasons[0]

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Match Prediction", "📊 Historical Analysis", "🏃 Player Stats", "📈 Team Analysis", "🕵️ Transfer Scout"])

    # --- TAB 1: PREDICTION (New Features) ---
    with tab1:
        st.header(f"Next Match Prediction ({current_season})")
        c_stats = create_team_features(df, season_filter=current_season)
        
        if c_stats.empty:
            st.warning("လက်ရှိရာသီအတွက် ဒေတာမရှိသေးပါ။")
        else:
            teams = sorted(c_stats['Team'].unique())
            col1, col2 = st.columns(2)
            h_team = col1.selectbox("Home Team", teams, key="h_team")
            a_team = col2.selectbox("Away Team", teams, index=1 if len(teams)>1 else 0, key="a_team")

            if st.button("Predict Match Outcome"):
                h_data = c_stats[c_stats['Team'] == h_team].iloc[0]
                a_data = c_stats[c_stats['Team'] == a_team].iloc[0]

                # Prediction Logic
                pred_h_goals = (h_data['HomeAttack'] + a_data['AwayDefense']) / 2
                pred_a_goals = (a_data['AwayAttack'] + h_data['HomeDefense']) / 2
                
                # Display Score
                st.markdown("### 🎯 Predicted Score")
                res_c1, res_c2, res_c3 = st.columns(3)
                res_c1.metric(h_team, f"{int(np.round(pred_h_goals))} Goals")
                res_c2.markdown("<h1 style='text-align: center;'>-</h1>", unsafe_allow_html=True)
                res_c3.metric(a_team, f"{int(np.round(pred_a_goals))} Goals")

                # Win Probability
                total_val = pred_h_goals + pred_a_goals
                h_win_p = round((pred_h_goals / total_val) * 100, 1)
                a_win_p = round((pred_a_goals / total_val) * 100, 1)

                st.markdown("### 📊 Win Probability")
                st.progress(h_win_p / 100, text=f"{h_team} Win Chance: {h_win_p}%")
                st.progress(a_win_p / 100, text=f"{a_team} Win Chance: {a_win_p}%")
                
                winner = h_team if pred_h_goals > pred_a_goals else (a_team if pred_a_goals > pred_h_goals else "Draw")
                if winner == "Draw":
                    st.info(f"💡 Prediction: This match is likely to end in a **Draw**.")
                else:
                    st.success(f"💡 Prediction: **{winner}** is the favorite to win!")

                # --- Player Goal Prediction ---
                st.divider()
                st.markdown("### ⚽ Top Goalscorer Prediction")
                
                with st.spinner("Calculating player probabilities..."):
                    p_df = scrape_player_stats("2024-25")
                
                if not p_df.empty:
                    # Function to calculate goal probability
                    def get_goal_prob(player, team_attack_strength, opp_defense_weakness):
                        try:
                            mins = float(player['Minutes'])
                            goals = float(player['Goals'])
                            form = float(player['Form'])
                            
                            if mins < 90: return 0.0
                            
                            # Base: Goals per 90 mins
                            goals_per_90 = (goals / mins) * 90
                            
                            # Adjust for form (Form is avg points per match recently)
                            form_factor = 1 + (form / 10.0)  # simple booster
                            
                            # Adjust for team strength context
                            # team_attack ~ 1.5 avg goals. opp_defense ~ 1.5 avg conceded.
                            match_factor = (team_attack_strength * opp_defense_weakness) / 2.5
                            
                            prob = goals_per_90 * form_factor * match_factor * 0.4 # 0.4 is a scaling constant
                            return min(prob, 0.99) # Max 99%
                        except:
                            return 0.0

                    # Prepare Home Players
                    h_players = p_df[p_df['Team'] == h_team].copy()
                    h_players['GoalProb'] = h_players.apply(
                        lambda x: get_goal_prob(x, h_data['HomeAttack'], a_data['AwayDefense']), axis=1
                    )
                    
                    # Prepare Away Players
                    a_players = p_df[p_df['Team'] == a_team].copy()
                    a_players['GoalProb'] = a_players.apply(
                        lambda x: get_goal_prob(x, a_data['AwayAttack'], h_data['HomeDefense']), axis=1
                    )
                    
                    # Display Side by Side
                    pc1, pc2 = st.columns(2)
                    
                    with pc1:
                        st.subheader(f"{h_team} Top Scorers")
                        top_h = h_players.sort_values('GoalProb', ascending=False).head(3)
                        for _, p in top_h.iterrows():
                            prob_pct = int(p['GoalProb'] * 100)
                            if prob_pct > 0:
                                st.progress(prob_pct / 100, text=f"**{p['Player Name']}**: {prob_pct}% to score")
                    
                    with pc2:
                        st.subheader(f"{a_team} Top Scorers")
                        top_a = a_players.sort_values('GoalProb', ascending=False).head(3)
                        for _, p in top_a.iterrows():
                            prob_pct = int(p['GoalProb'] * 100)
                            if prob_pct > 0:
                                st.progress(prob_pct / 100, text=f"**{p['Player Name']}**: {prob_pct}% to score")

    # --- TAB 2: ANALYSIS (Old Code Restored) ---
    with tab2:
        st.header("Previous Season Analysis")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            sel_season = st.selectbox("Select Season", all_seasons, key="sel_s")
        with col_s2:
            season_teams = sorted(df[df['Season'] == sel_season]['HomeTeam'].unique())
            sel_team = st.selectbox("Select Team", season_teams, key="sel_t")

        analysis_df = df[((df['HomeTeam'] == sel_team) | (df['AwayTeam'] == sel_team)) & (df['Season'] == sel_season)].copy()

        if not analysis_df.empty:
            analysis_df['Goals Scored'] = analysis_df.apply(lambda x: x['FTHG'] if x['HomeTeam'] == sel_team else x['FTAG'], axis=1)
            analysis_df['Goals Conceded'] = analysis_df.apply(lambda x: x['FTAG'] if x['HomeTeam'] == sel_team else x['FTHG'], axis=1)
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Matches", len(analysis_df))
            m2.metric("Avg Scored", round(analysis_df['Goals Scored'].mean(), 2))
            m3.metric("Avg Conceded", round(analysis_df['Goals Conceded'].mean(), 2))
            
            st.divider()
            st.subheader(f"{sel_team} Performance Graph")
            st.bar_chart(analysis_df[['Goals Scored', 'Goals Conceded']])
            
            with st.expander("Show Detailed Match Results"):
                st.dataframe(analysis_df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']], use_container_width=True)

    # --- TAB 3: PLAYER STATS (New) ---
    with tab3:
        st.header("🏃 Player Statistics")
        st.markdown("Data sourced from FPL (Fantasy Premier League).")
        
        # Season Selection for Player Stats
        available_player_seasons = ["2024-25", "2023-24", "2022-23", "2021-22", "2020-21"]
        sel_player_season = st.selectbox("Select Season for Stats", available_player_seasons, key="player_season")
        
        with st.spinner(f"Fetching player data for {sel_player_season}..."):
            player_df = scrape_player_stats(sel_player_season)
        
        if not player_df.empty:
            # Filters
            col_f1, col_f2 = st.columns(2)
            search_term = col_f1.text_input("Search Player", "")
            
            # Filter by name
            if search_term:
                player_df = player_df[player_df['Player Name'].str.contains(search_term, case=False)]
                
            # Top Performers
            st.subheader("Top Performers")
            st.dataframe(player_df, use_container_width=True, height=600)
            
            # Quick Stats
            st.divider()
            qs1, qs2, qs3 = st.columns(3)
            top_scorer = player_df.sort_values('Goals', ascending=False).iloc[0]
            top_assister = player_df.sort_values('Assists', ascending=False).iloc[0]
            most_points = player_df.sort_values('Total Points', ascending=False).iloc[0]
            
            qs1.metric("⚽ Top Scorer", f"{top_scorer['Player Name']} ({top_scorer['Goals']})")
            qs2.metric("🅰️ Most Assists", f"{top_assister['Player Name']} ({top_assister['Assists']})")
            qs3.metric("⭐ MVP (FPL Points)", f"{most_points['Player Name']} ({most_points['Total Points']})")
            
        else:
            st.error("Failed to load player statistics. Please check your internet connection.")

    # --- TAB 4: TEAM ANALYSIS (New) ---
    with tab4:
        st.header("📈 Team & Squad Analysis")
        
        # 1. Team Selection
        all_teams_hist = sorted(df['HomeTeam'].unique())
        t_sel = st.selectbox("Select Team for Analysis", all_teams_hist, key="t_analysis")
        
        # 2. Historical Performance (Win Rate & Form)
        # Filter matches involving this team
        t_matches = df[((df['HomeTeam'] == t_sel) | (df['AwayTeam'] == t_sel)) & (df['Season'] == current_season)].copy()
        
        if not t_matches.empty:
            t_matches['Result'] = t_matches.apply(
                lambda x: 'W' if (x['HomeTeam'] == t_sel and x['FTHG'] > x['FTAG']) or (x['AwayTeam'] == t_sel and x['FTAG'] > x['FTHG'])
                else ('D' if x['FTHG'] == x['FTAG'] else 'L'), axis=1
            )
            
            wins = len(t_matches[t_matches['Result'] == 'W'])
            draws = len(t_matches[t_matches['Result'] == 'D'])
            losses = len(t_matches[t_matches['Result'] == 'L'])
            total = len(t_matches)
            win_rate = (wins / total) * 100
            
            # Form Guide (Last 5)
            form_str = " - ".join(t_matches.sort_values('Date')['Result'].tail(5).tolist())
            
            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Win Rate", f"{win_rate:.1f}%")
            c2.metric("Wins", wins)
            c3.metric("Draws", draws)
            c4.metric("Losses", losses)
            
            st.caption(f"Recent Form (Last 5 Matches): **{form_str}**")
            
            # Goals Chart
            st.subheader("Goals Scored vs Conceded (Season Progress)")
            t_matches['GoalsFor'] = t_matches.apply(lambda x: x['FTHG'] if x['HomeTeam'] == t_sel else x['FTAG'], axis=1)
            t_matches['GoalsAgainst'] = t_matches.apply(lambda x: x['FTAG'] if x['HomeTeam'] == t_sel else x['FTHG'], axis=1)
            t_matches['Matchday'] = range(1, len(t_matches) + 1)
            
            st.line_chart(t_matches.set_index('Matchday')[['GoalsFor', 'GoalsAgainst']])
            
        else:
            st.warning(f"No match data found for {t_sel} in {current_season}.")

        st.divider()
        
        # 3. Squad Analysis (Player Stats)
        st.subheader(f"🏃 {t_sel} Squad Performance")
        
        with st.spinner("Loading squad data..."):
            sq_df = scrape_player_stats("2024-25")
            
        if not sq_df.empty:
            # Filter for selected team
            # Note: scrape_player_stats uses mapped names like "Man City", user selection from df uses "Man City"
            # We verified this mapping in previous steps.
            team_squad = sq_df[sq_df['Team'] == t_sel]
            
            if not team_squad.empty:
                # Key Stats Table
                st.dataframe(
                    team_squad[['Player Name', 'Position ID', 'Goals', 'Assists', 'Total Points', 'Price', 'Minutes', 'Form']],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualizations
                st.markdown("### 📊 Performance Visualizations")
                
                vc1, vc2 = st.columns(2)
                
                with vc1:
                    st.markdown("**Top Goalscorers**")
                    top_scorers = team_squad[team_squad['Goals'] > 0].sort_values('Goals', ascending=False).head(10)
                    st.bar_chart(top_scorers.set_index('Player Name')['Goals'])
                    
                with vc2:
                    st.markdown("**Value for Money (Price vs Points)**")
                    # Simple Scatter using Streamlit
                    st.scatter_chart(
                        team_squad,
                        x='Price',
                        y='Total Points',
                        color='Position ID',
                        size='Minutes',  # Bubble size by minutes played
                            use_container_width=True
                    )
                    st.caption("Bubble size represents minutes played. Color represents Position.")
                    
            else:
                st.info(f"Could not match squad data for '{t_sel}'. Check if Team Name mapping in scraper.py matches historical data.")

    # --- TAB 5: TRANSFER SCOUT (New) ---
    # --- TAB 5: TRANSFER SCOUT (New) ---
    with tab5:
        st.header("🕵️ AI Transfer Scout")
        st.markdown("Data-driven buy recommendations based on Form, ICT Index, and Value.")
        
        with st.spinner("Analyzing player market..."):
            market_df = scrape_player_stats("2024-25")
            
        if not market_df.empty:
            # 1. Team Context & Filters
            st.subheader("🛠️ Scout Settings")
            c_ctx1, c_ctx2 = st.columns(2)
            
            # User's Team Selection
            all_teams = sorted(market_df['Team'].unique())
            my_team = c_ctx1.selectbox("Your Managed Team", all_teams, index=0, key="my_team_select")
            
            # Position Filter
            pos_map = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
            market_df['Position'] = market_df['Position ID'].map(pos_map)
            sel_pos = c_ctx2.selectbox("Target Position", ["Goalkeeper", "Defender", "Midfielder", "Forward"], index=3)
            
            col_ts1, col_ts2 = st.columns(2)
            # Budget Filter
            max_price = col_ts1.slider("Max Budget (£m)", 4.0, 15.0, 8.5, 0.1)
            
            # Exclude Team (Optional - e.g. Rivals)
            exclude_team = col_ts2.text_input("Exclude Other Team (Optional)", "")
            
            st.divider()

            # 2. Logic: Market vs My Team
            # My Current Best Option
            my_squad = market_df[(market_df['Team'] == my_team) & (market_df['Position'] == sel_pos)]
            my_best_player = None
            if not my_squad.empty:
                my_best_player = my_squad.sort_values('Total Points', ascending=False).iloc[0]

            # Market Options (Excluding My Team)
            scout_df = market_df[(market_df['Team'] != my_team) & (market_df['Price'] <= max_price)].copy()
            if sel_pos != "All":
                scout_df = scout_df[scout_df['Position'] == sel_pos]
            if exclude_team:
                scout_df = scout_df[~scout_df['Team'].astype(str).str.contains(exclude_team, case=False)]
                
            # Recommendation Engine (Scout Score)
            # Weighted Score: 40% Form, 40% ICT Index, 20% Value (Points/Price)
            def normalize(series):
                return (series - series.min()) / (series.max() - series.min())
            
            if not scout_df.empty:
                scout_df['Norm_Form'] = normalize(scout_df['Form'])
                scout_df['Norm_ICT'] = normalize(scout_df['ICT Index'])
                scout_df['PointsPerPrice'] = scout_df['Total Points'] / scout_df['Price']
                scout_df['Norm_Value'] = normalize(scout_df['PointsPerPrice'])
                
                scout_df['Scout Score'] = (
                    (scout_df['Norm_Form'] * 0.4) + 
                    (scout_df['Norm_ICT'] * 0.4) + 
                    (scout_df['Norm_Value'] * 0.2)
                ) * 100
                
                # Get Top 5 Recommendations
                top_targets = scout_df.sort_values('Scout Score', ascending=False).head(5)
                
                st.subheader(f"🏆 Top Recommendations vs Your Squad")
                
                if my_best_player is not None:
                    st.info(f"ℹ️ Your current best {sel_pos} is **{my_best_player['Player Name']}** ({my_best_player['Total Points']} pts).")
                
                # Display Top Targets
                for i, row in top_targets.iterrows():
                    with st.expander(f"#{list(top_targets.index).index(i)+1}: {row['Player Name']} ({row['Team']}) - £{row['Price']}m", expanded=(i==top_targets.index[0])):
                        c_a, c_b = st.columns([1, 2])
                        
                        with c_a:
                            st.metric("Scout Score", f"{row['Scout Score']:.1f}/100")
                            st.metric("Form", row['Form'], delta=round(row['Form'] - my_best_player['Form'], 1) if my_best_player is not None else None)
                            st.metric("ICT Index", f"{row['ICT Index']:.1f}", delta=round(row['ICT Index'] - my_best_player['ICT Index'], 1) if my_best_player is not None else None)
                            
                        with c_b:
                            st.markdown("#### 💡 Upgrade Analysis")
                            reasons = []
                            if row['Norm_Form'] > 0.8: reasons.append("🔥 **In exceptional form** recently.")
                            elif row['Norm_Form'] > 0.6: reasons.append("📈 **Good consistent form**.")
                            
                            if row['Norm_ICT'] > 0.8: reasons.append("🧠 **High Creativity & Threat** creator.")
                            
                            if row['PointsPerPrice'] > scout_df['PointsPerPrice'].quantile(0.75):
                                reasons.append("💰 **Great Value**: High points return for this price.")
                                
                            if row['ICT Index'] > 100:
                                reasons.append("⚡ **Impact Player**: High influence on matches.")
                            
                            # Comparative Logic
                            if my_best_player is not None:
                                if row['Total Points'] > my_best_player['Total Points']:
                                    reasons.append(f"🏆 **Better Season**: Has {row['Total Points'] - my_best_player['Total Points']} more points than {my_best_player['Player Name']}.")
                                if row['Form'] > my_best_player['Form']:
                                    reasons.append(f"🚀 **Better Form**: currently outperforming {my_best_player['Player Name']}.")

                            for r in reasons:
                                st.markdown(f"- {r}")
                                
                            # Comparison Chart (Radar proxy via Bar)
                            plot_data = pd.DataFrame({
                                'Metric': ['Form', 'Influence', 'Creativity', 'Threat'],
                                'Target': [float(row['Form']), float(row['Influence']), float(row['Creativity']), float(row['Threat'])]
                            })
                            if my_best_player is not None:
                                plot_data['Your Player'] = [float(my_best_player['Form']), float(my_best_player['Influence']), float(my_best_player['Creativity']), float(my_best_player['Threat'])]
                            
                            st.bar_chart(plot_data.set_index('Metric'))

                st.divider()
                
                # 3. Comparative Charts
                st.subheader("📊 Market Analysis")
                
                tc1, tc2 = st.columns(2)
                
                with tc1:
                    st.markdown("**ICT Index vs Price (Influence, Creativity, Threat)**")
                    st.scatter_chart(
                        scout_df,
                        x='Price',
                        y='ICT Index',
                        color='Team',
                        size='Scout Score',
                        use_container_width=True
                    )
                    st.caption("Identify high-impact players (High ICT) at lower prices.")
                    
                with tc2:
                    st.markdown("**Form vs Value (ROI)**")
                    st.scatter_chart(
                        scout_df,
                        x='Price',
                        y='Form',
                        color='Position',
                        size='Total Points',
                        use_container_width=True
                    )
                    st.caption("Identify in-form players relative to their cost.")

            else:
                st.warning("No players found matching these criteria. Try increasing the budget.")