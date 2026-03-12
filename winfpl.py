import streamlit as st
from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb
import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split  
import plotly.express as px
import plotly.graph_objects as go

# Show ALL columns/rows 
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)       # Prevent line wrapping
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 50)  # Truncate long text

# Streamlit Dashboard

# 🚀 Pro Config + Custom CSS
st.set_page_config(
    page_title="⚽🏆WinFPL - FPL Pro Analytics/ML Predictions", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding: 2rem;}
    .metric-container {text-align: center; font-size: 1.2rem;}
    .stPlotlyChart {height: 400px !important;}
    </style>
""", unsafe_allow_html=True)






@st.cache_data()   #ttl=600  #10 min refresh
def fetch_fpl_data():
    """
    SINGLE FUNCTION: Scrapes FPL players, teams, AND histories DataFrames.
    Returns: (players, teams, histories)
    """
    print("🔄 Fetching bootstrap data...")
    
    # 1. BOOTSTRAP DATA (players + teams)
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url, verify=False, timeout=10).json()
    
    # 2. PLAYERS (active only)
    players = pd.json_normalize(data['elements'])
    players = players[players['status'] == 'a'].copy()
    print(f"📊 Found {len(players)} active players")
    

    # 3. TEAMS
    teams = pd.json_normalize(data['teams'])
    team_map = teams.set_index('id')['name'].to_dict()
    players['team_name'] = players['team'].map(team_map)
    
    # 4. BUSINESS METRICS (FIXED per-player PPG)
    #players['price'] = players['now_cost'] / 10
    players['points_per_game'] = (players['total_points'] / np.maximum(players['starts'], 1)).round(1)
    players['points_per_game'] = pd.to_numeric(players['points_per_game'], errors='coerce')
    
    # 5. POSITIONS
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    players['Position'] = players['element_type'].map(pos_map)
    
    # 6. NUMERIC COLUMNS
    numeric_cols = ['form', 'total_points', 'dreamteam_count', 'minutes', 'selected', 'selected_by_percent', 'value_season', 'goals_scored', 'assists', 'goals_conceded', 'yellow_cards', 'red_cards', 'bonus', 'bps', 'price', 'influence', 'creativity', 'threat', 'ict_index', 'defensive_contribution', 'starts']
    for col in numeric_cols:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors='coerce')
                
    
    # 7. HISTORIES (TOP 50 players for speed)
    print("📈 Fetching histories (top 50 players)...")
    top_player_ids = players.nlargest(500, 'form')['id'].tolist()  # Best form players
    histories = fetch_player_histories(top_player_ids)
    histories = histories.rename(columns={'total_points': 'gw_points'})
    print(f"📋 Histories shape: {histories.shape}")
    
    return players, teams, histories



def fetch_player_histories(player_ids):
    """Robust history fetcher with retries"""
    session = requests.Session()
    retry_strategy = Retry(total=4, backoff_factor=1, 
                          status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    
    histories = []
    for i, pid in enumerate(player_ids):
        try:
            url = f'https://fantasy.premierleague.com/api/element-summary/{pid}/'
            resp = session.get(url, verify=False, timeout=8)
            resp.raise_for_status()
            
            data = resp.json()
            hist = data.get('history', [])[:]  # Last 5 gameweeks = -5              # : = all gameweeks

            
            if hist:
                hist_df = pd.json_normalize(hist)
                hist_df['player_id'] = pid
                histories.append(hist_df)
                print(f"✅ Player {pid}: {len(hist)} gameweeks")
            
            time.sleep(0.3)  # Rate limit
        except Exception as e:
            print(f"❌ Player {pid}: {str(e)[:]}")    #Change : with the number of players you want to include like- (:50)
            continue
    
    if histories:
        histories = pd.concat(histories, ignore_index=True)   
        return histories
    return pd.DataFrame()



# Run the scraper    
    
def preprocess_fpl_data(players, teams, histories):
    """
    SINGLE FUNCTION: Preprocess + Feature Engineering + Merge ALL DataFrames
    Returns: master_df ready for XGBoost
    """
    print("🔧 Preprocessing data...")
    
    # 1. CLEAN PLAYERS
    players = players.copy()
    players['Name'] = players['first_name'] + ' ' + players['second_name']
    players['price'] = pd.to_numeric(players['now_cost'] / 10, errors='coerce')
    
    # Age calculation (today - birth_date)
    players['birth_date'] = pd.to_datetime(players['birth_date'], errors='coerce')
    today = pd.to_datetime(date.today())
    players['age'] = ((today - players['birth_date']).dt.days / 365.25).round(0)

    
    # Value metrics
    players['points_per_million'] = (players['total_points'] / players['price']).round(0)
    players['minutes_per_game'] = (np.minimum(90, players['minutes'] / np.maximum(players['starts'], 0.1))).round(0)          
        
        
    # OPPONENT FIXTURE DIFFICULTY 
    print("⚽ Adding opponent fixture difficulty...")
    try:
        fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
        fixtures = requests.get(fixtures_url, verify=False, timeout=10).json()
    
        # SAFEST upcoming filter
        upcoming = []
        for f in fixtures:
            finished = f.get('finished', True)
            event = f.get('event')
            if not finished and event is not None:
                upcoming.append(f)
    
        # BULLETPROOF max()
        if upcoming:
            next_gw = max(f['event'] for f in upcoming)
            print(f"✅ Next GW: {next_gw}")
        else:
            next_gw = None
            print("⚠️ No upcoming fixtures - using team strength")
    
        # Build opponent mapping 
        next_opponents = {}
        if next_gw:
            for f in fixtures:
                if f['event'] == next_gw:
                    next_opponents[f['team_h']] = f['team_a']
                    next_opponents[f['team_a']] = f['team_h']
    
        # Default to own team rank if no fixtures
        players['opp_team'] = players['team'].map(next_opponents).fillna(players['team'])
    
        # Merge OPPONENT rank 
        players = players.merge(teams[['id', 'position']], left_on='opp_team', right_on='id', how='left')
        players = players.rename(columns={'position': 'opp_team_rank'})
    
        # Handle missing ranks
        players['opp_team_rank'] = players['opp_team_rank'].fillna(10.5).astype(float)
        players['fixture_difficulty'] = players['opp_team_rank'] / 20  # 0-1 scale
    
        print(f"Fixture range: {players['fixture_difficulty'].min():.2f}-{players['fixture_difficulty'].max():.2f}")
    
    except Exception as e:
        print(f"⚠️ Fixture fallback: {e}")
        players['fixture_difficulty'] = 3.0 / 20  # 0.15 neutral    

        
    players['fixture_difficulty'] = pd.to_numeric(players['fixture_difficulty'], errors='coerce').fillna(0.15)





    # 2. HISTORY FEATURES (if histories exist)
    if not histories.empty:
        print("📈 Creating history features...")
               
        
        # Numeric history columns
        hist_cols = ['gw_points', 'minutes', 'round', 'form', 'goals_scored', 'assists', 'goals_conceded', 'yellow_cards', 'red_cards', 'bonus', 'bps', 'price', 'influence', 'creativity', 'threat', 'ict_index', 'defensive_contribution', 'expected_goals', 'expected_assists', 'transfers_in', 'transfers_out']
        for col in hist_cols:
            if col in histories.columns:
                histories[col] = pd.to_numeric(histories[col], errors='coerce')
        
        # RECENT FORM (last 3 gameweeks average)
        histories = histories.sort_values(['player_id', 'round'])
        def get_recent_points(player_id, histories):  # Pass histories explicitly
            """Calculate recent form for FPL player"""
            player_history = histories[histories['player_id'] == player_id]
            if len(player_history) >= 3:
                return player_history['gw_points'].tail(3).mean()
            return player_history['gw_points'].mean() or 0  # Handle no history

        players['3gw_points'] = players['id_x'].apply(
            lambda pid: get_recent_points(pid, histories)).fillna(0).round(1)

        
        # FORM TREND (point change over last 3 games)
        histories['point_change'] = histories.groupby('player_id')['gw_points'].diff()
        
        def safe_form_trend(player_id):
            player_history = histories[histories['element'] == player_id]
            if len(player_history) >= 3:
                recent_3 = player_history['gw_points'].tail(3).mean()
                prev_3 = player_history['gw_points'].iloc[-6:-3].mean() if len(player_history) >= 6 else 0
                return (recent_3 - prev_3) / 10  # Normalized trend
            return 0
        
        # AVG MINUTES (last 5 games)
        players['recent_minutes'] = players['id_x'].apply(
            lambda pid: histories[histories['element'] == pid]['minutes'].tail(5).mean()
            if len(histories[histories['element']==pid])>= 0 else players[players['id_x']==pid]['minutes'].iloc[0]).fillna(players['minutes'])
    
    else:
        # Fallback if no histories
        players['3gw_points'] = players['form']
        players['form_trend'] = 0
        players['recent_minutes'] = players['minutes']
    
    # 3. SELECTING FEATURES FOR ML
    feature_cols = [
        'form', 'points_per_million', 'points_per_game', '3gw_points', 
        'form_trend', 'recent_minutes', 'ict_index', 'selected', 'fixture_difficulty',
        'dreamteam_count', 'minutes_per_game', 'point_change', 'total_points', 'age'
    ]
    
    # Only keep available columns
    available_features = [col for col in feature_cols if col in players.columns]
    print(f"✅ Using features: {available_features}")
    
    
    #For checking the data that we are scraping    
    


    
    # 4. MERGE EVERYTHING → MASTER DATAFRAME    
    master_df = pd.merge(histories[['element', 'gw_points', 'minutes', 'round', 'goals_scored', 'assists', 'goals_conceded', 'yellow_cards', 'red_cards', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'defensive_contribution', 'expected_goals', 'expected_assists', 'transfers_in', 'transfers_out']],   
                         players[['id_x', 'Name', 'web_name', 'team_name', 'Position', 'price'] + available_features].copy(), left_on= "element", right_on= "id_x", how= "left")
    
    
    # Handle NaN
    master_df = master_df.fillna(0)
    
    #Feature Engineering
    # 1. Sort by player + gameweek (CRITICAL)
    master_df = master_df.sort_values(['id_x', 'round']).reset_index(drop=True)
    print(master_df.shape)
    
    # 2. Create target: next GW points
    master_df['target'] = master_df.groupby('id_x')['gw_points'].shift(-1)
    print(master_df.shape)
    print(master_df['target'].max())
    
    df_ml = master_df     #.dropna(subset=['target'])
    print(df_ml.shape)
    print(df_ml['target'].max())
    print(df_ml['round'].max())
    print(df_ml['round'].min())
    
    # 4. Features from HISTORY (past 5 GWs)
    df_ml['form_5gw'] = df_ml.groupby('id_x')['total_points'].rolling(5, min_periods=1).mean().reset_index(0, drop=True).shift(1)
    df_ml['influence_5gw'] = df_ml.groupby('element')['influence'].rolling(5, min_periods=1).mean().reset_index(0, drop=True).shift(1)
    df_ml['ictindex_5gw'] = df_ml.groupby('element')['ict_index'].rolling(5, min_periods=1).mean().reset_index(0, drop=True).shift(1)
    #Bonus Points Momentum
    df_ml['bonus_5gw'] = (df_ml.groupby('element')['bonus'].rolling(5, min_periods=1).mean().reset_index(0, drop=True).shift(1))
    df_ml['bps_5gw'] = (df_ml.groupby('element')['bps'].rolling(5, min_periods=1).mean().reset_index(0, drop=True).shift(1))
    print(df_ml.shape)
    df_ml = df_ml[['id_x', 'Name', 'web_name', 'team_name', 'Position',  'age', 'dreamteam_count', 'price', 'gw_points', 'total_points','form', 'points_per_million', 'points_per_game', '3gw_points', 'ict_index', 'minutes_per_game', 'minutes', 'round', 'goals_scored', 'assists', 'goals_conceded', 'yellow_cards', 'red_cards', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'defensive_contribution', 'form_5gw', 'influence_5gw', 'ictindex_5gw', 'bonus_5gw', 'bps_5gw', 'transfers_in', 'transfers_out', 'fixture_difficulty', 'target']]
    print(df_ml.shape)
    print(df_ml['target'].max())
    print(df_ml['round'].max())
    print(df_ml['round'].min())
    #Saving different dataframe for making model as well as saving this for making predictions
    # 5. Drop last GW per player (no target)
    master_df = df_ml.dropna(subset=['target'])
    print(master_df.shape)
    print(master_df['target'].max())
    print(master_df['round'].max())
    print(master_df['round'].min())
    #'selected','form_trend','point_change','expected_goals', 'expected_assists',
    df_ml = df_ml[df_ml['round'] == df_ml['round'].max()]
    print(df_ml.shape)
    print(df_ml['target'].max())
    print(df_ml['round'].max())
    print(df_ml['round'].min())
    
    
    #print(f"🎯 Master DataFrame: {master_df.shape}")
    #print("Sample:")
    return df_ml, master_df
    

#Saving different dataframe for making model as well as saving this for making predictions


# RUN EVERYTHING
# USAGE - Complete Pipeline
def create_ml_pipeline():

    """FPL ML Pipeline - Returns train/test splits"""
    players, teams, histories = fetch_fpl_data()
    df_ml, master_df = preprocess_fpl_data(players, teams, histories)

    # SAFE VERSION:
    feature_cols = ['age', 'dreamteam_count', 'price', 'total_points','form', 'points_per_million', 'points_per_game', '3gw_points', 'ict_index', 'minutes_per_game', 'minutes', 'goals_scored', 'assists', 'goals_conceded', 'yellow_cards', 'red_cards', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'form_5gw', 'influence_5gw', 'ictindex_5gw', 'bonus_5gw', 'bps_5gw', 'transfers_in', 'transfers_out', 'fixture_difficulty']           #'gw_points',
    available_cols = []

    # Check each column individually (handles ANY data type)
    for col in feature_cols:
        if isinstance(col, str) and col in master_df.columns:  # ← STRING CHECK
            available_cols.append(col)

    print(f"✅ Available ML features: {available_cols}")  
    
    
    X = master_df[available_cols].fillna(0)
    y = master_df['target'].fillna(0)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, master_df, df_ml  # ← CRITICAL RETURN

# Step 3. ML READY MATRICES

print(f"✅ ML Pipeline Complete!")
#print(f"Features: {feature_cols}")
#print(f"X_train shape: {X_train.shape} | y_train: {y_train.shape}")
    

# ONE LINE EXECUTION
if __name__ == "__main__":
    
    X_train, X_test, y_train, y_test, master_df, df_ml = create_ml_pipeline()
    
    print("🎉 SUCCESS!")
    print(f"X_train: {X_train.shape}, Master: {master_df.shape}")

model = RandomForestRegressor(n_estimators=50, n_jobs=1, random_state=42)
model.fit(X_train, y_train)

master_df['predicted_points'] = model.predict(master_df[X_test.columns].fillna(0))
print("🏆 TOP FPL PICKS:")
print(master_df.nlargest(10, 'predicted_points')[['web_name', 'predicted_points']])

df_ml['predicted_points'] = model.predict(df_ml[X_test.columns].fillna(0))
print("🏆 TOP FPL PICKS:")
print(df_ml.nlargest(10, 'predicted_points')[['web_name', 'predicted_points']])

 

 

#Streamlit continues...

# Hero Header
st.title("⚽🏆WinFPL - FPL Pro Analytics/ML Predictions")
#st.markdown("---")
st.success(f"✅ Loaded {len(df_ml)} players | Last updated: {pd.Timestamp.now().strftime('%H:%M')}")


# 🔥 HERO KPIs ROW (NEW!)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Points", f"{df_ml['total_points'].sum():,.0f}", delta="↑ Live")
with col2:
    top_ppm_player = df_ml.nlargest(1, 'points_per_game')
    st.metric("PPM Leader", f"{top_ppm_player['points_per_game'].values[0]:.2f}", top_ppm_player['Name'].values[0])
with col3:
    top_form_player = df_ml.nlargest(1, 'form')
    st.metric("Form", f"{top_form_player['form'].iloc[0]:.1f}", top_form_player['Name'].iloc[0])
with col4:
    # Get top points getter this week (player name)
    top_gw_player = df_ml.nlargest(1, 'gw_points')
    st.metric("Current_GW_ Points", f"{top_gw_player['gw_points'].iloc[0]:.1f}", top_gw_player['Name'].iloc[0])
with col5:
    # Get the name of top points predicted this week
    top_pred_player = df_ml.nlargest(1, 'predicted_points')
    st.metric("Predicted_ Points", f"{top_pred_player['predicted_points'].iloc[0]:.1f}", top_pred_player['Name'].iloc[0])


st.markdown("---")

# 🔍 ENHANCED FILTERS ROW
colA, colB = st.columns([2, 1])
with colA:
    search = st.text_input("🔍 Search Players", placeholder="Enter player name...")

# Sidebar Power Filters
st.sidebar.header("⚙️ Advanced Filters")
positions = st.sidebar.multiselect("Position", df_ml['Position'].unique(), default=df_ml['Position'].unique())
team_filter = st.sidebar.multiselect("Team", df_ml['team_name'].unique(), default=df_ml['team_name'].unique()[:])
min_points = st.sidebar.slider("Min Points", int(df_ml['total_points'].min()), int(df_ml['total_points'].max()), int(df_ml['total_points'].min()))
Age = st.sidebar.slider("Age", int(df_ml['age'].min()), int(df_ml['age'].max()), int(df_ml['age'].min()))
min_pred = st.sidebar.slider("Min Predicted", int(df_ml['predicted_points'].min()), int(df_ml['predicted_points'].max()), int(df_ml['predicted_points'].min()))
price_range = st.sidebar.slider("Price Range", 4.0, 15.0, (4.0, 15.0))


# Apply Filters (Fixed Logic)
filtered = df_ml[
    (df_ml['Position'].isin(positions)) &
    (df_ml['team_name'].isin(team_filter)) &
    (df_ml['total_points'] >= min_points) &
    (df_ml['age'] >= Age) &
    (df_ml['predicted_points'] >= min_pred) &
    (df_ml['price'] >= price_range[0]) & (df_ml['price'] <= price_range[1])
]



if search:
    filtered = filtered[filtered['Name'].str.contains(search, case=False, na=False)]

st.info(f"📊 Showing {len(filtered)}/{len(df_ml)} players")

# 📊 RESPONSIVE 2-ROW LAYOUT
row1_col1, row1_col2 = st.columns([2, 1])
row2_col1, row2_col2 = st.columns([2, 1])
row3_col1, row3_col2 = st.columns([2,1])
row4_col1, row4_col2 = st.columns([2,1])

# Top Left: Pro Player Table
with row1_col1:
    st.subheader("🏆 Top Predicted Players")
    top_players = filtered[['Name', 'team_name', 'Position', 'dreamteam_count', 'price', 'gw_points', 'total_points', 'form', 'round', 'goals_scored', 'assists', 'bonus', 'bps', 'defensive_contribution', 'transfers_in', 'transfers_out','fixture_difficulty', 'predicted_points']].sort_values('predicted_points', ascending=False)
    
    # Pro styling
    styled_df = top_players
    st.dataframe(styled_df, use_container_width=True, height=450, hide_index=True)
  
  
  
    
# Top Right: Value Chart
with row1_col2:
    st.subheader("Total Points vs 💰Price")
    fig_value = px.scatter(
        filtered.head(200), x='price', y='total_points', 
        size='form', color='predicted_points', hover_name='Name',
        color_continuous_scale='RdYlGn',
        title=None
    )
    fig_value.update_layout(height=450)
    st.plotly_chart(fig_value, use_container_width=True)

# Bottom Left: Form vs Predicted
with row2_col1:
    st.subheader("📈 Form vs Predicted")
    fig_form = px.scatter(
        filtered, x='form', y='predicted_points', 
        size='price', color='Position', hover_name='Name',
        category_orders={'Position': ['GK', 'DEF', 'MID', 'FWD']}
    )
    fig_form.update_layout(height=450)
    st.plotly_chart(fig_form, use_container_width=True)

# Bottom Right: Team Predictions
with row2_col2:
    st.subheader("Total Predicted Points of each team🔮")
    team_pred = filtered.groupby('team_name')['predicted_points'].sum().sort_values(ascending=False)
    fig_team = px.bar(
        x=team_pred.index, y=team_pred.values,
        color=team_pred.values, color_continuous_scale='Viridis',
        text=team_pred.values.round(0),
        title=None
    )
    fig_team.update_layout(height=450)
    st.plotly_chart(fig_team, use_container_width=True)
    
    
# Bottom Left: Goals Scored/Assists
with row3_col1:
    st.subheader("Total goals/assists scored this week by players in each team")
    fig = px.bar(filtered, x= 'team_name', y= 'goals_scored', color= 'assists', color_continuous_scale='RdYlGn', text= 'Name')  #, hover_name= ' Name'
    st.plotly_chart(fig, use_container_width=True)

# Bottom right: Points Scored
with row3_col2:
    st.subheader("Total points earned by players in each team")
    fig = px.bar(filtered, x = 'team_name', y = 'total_points', color = 'Name', text = 'Name')
    st.plotly_chart(fig, use_container_width=True)
    
    
# Bottom Left: Points Scored from each position
with row4_col1: 
    st.subheader("Total points earned based on the playing positon of each team")
    fig = px.bar(filtered, x='team_name',  y= 'total_points', color = 'Position', color_discrete_sequence=px.colors.qualitative.Set3, text ='Name')
    st.plotly_chart(fig, use_container_width=True)
    
    
# Bottom right: Goals Scored from each position
with row4_col2:
    st.subheader("Number of goals scored by each team based on the playing position")
    fig = px.bar(filtered, x='Position',  y= 'goals_scored' , color = 'Name', text ='team_name')
    st.plotly_chart(fig, use_container_width=True)



st.subheader("Max points earned by players in each team")
leaders = filtered.nlargest(20, 'total_points')
fig_cap = px.treemap(
        leaders, path=['Name'], values='total_points',
        color='form', hover_data=['team_name', 'price'],
        color_continuous_scale='Viridis',
)
st.plotly_chart(fig_cap, use_container_width=True)

    


# 🎯 CAPTAIN RECOMMENDATIONS (NEW!)
st.markdown("---")
st.subheader("👑 Captaincy Heatmap")
captains = filtered.nlargest(10, 'predicted_points')
fig_cap = px.treemap(
    captains, path=['Name'], values='predicted_points',
    color='form', hover_data=['team_name', 'price'],
    color_continuous_scale='RdYlGn'
)
st.plotly_chart(fig_cap, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ for FPL managers*")     #| Auto-refreshes every 10 mins

   
    
    

#line 86: change to 500
#line 111: change to :-5
#line 122: with the number of players you want to include like- (:50)



#conda activate fantasy   
#cd C:\Users\jwels\scrap

#streamlit run winfpl.py
