import os
import importlib
import re
import time
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import streamlit as st

st.set_page_config(page_title="Snake Master Draft Assistant", layout="wide")
UNAVAILABLE_PLAYERS = {
    'Brandon Aiyuk',
    'Josh Jacobs',
}
st.title("🐍 Reasoning Snake Draft Assistant (0.5 PPR)")

# ---------------------------------------------------------
# 1. SETUP LEAGUE & SNAKE CONFIGURATION
# ---------------------------------------------------------
st.sidebar.header("⚙️ Draft Setup")
num_teams = st.sidebar.number_input(
    "Number of Teams in League", min_value=8, max_value=16, value=12
)
my_pick_pos = st.sidebar.number_input(
    "Your Draft Slot (1-12)", min_value=1, max_value=num_teams, value=1
)

st.sidebar.header("🔗 ESPN Live Sync")
espn_league_id = st.sidebar.text_input("ESPN League ID", key="espn_league_id")
espn_season = st.sidebar.number_input(
    "ESPN Season", min_value=2020, max_value=2035, value=2026,
    key="espn_season",
)
espn_s2 = st.sidebar.text_input(
    "ESPN_S2 cookie", type="password", key="espn_s2",
    help="Copy the espn_s2 cookie value from your ESPN browser session.",
)
espn_swid = st.sidebar.text_input(
    "SWID cookie", type="password", key="espn_swid",
    help="Copy the SWID cookie value, including braces, from your ESPN browser session.",
)
espn_auto_sync = st.sidebar.checkbox("Live sync", value=True, key="espn_auto_sync")
espn_sync_seconds = st.sidebar.slider(
    "Sync interval (seconds)", min_value=3, max_value=30, value=5,
    key="espn_sync_seconds",
)

st.sidebar.header("⚖️ Value Adjustments")

# Dynamic Slider for Rookie Upside
rookie_boost_pct = st.sidebar.slider(
    "Rookie Upside Boost (%)",
    min_value=0,
    max_value=25,
    value=5,
    step=1,
    help="Increases 2026 projections for rookies to account for unmapped ceiling."
)

# Multiplier Conversion
rookie_mult = 1.0 + (rookie_boost_pct / 100.0)

ROSTER_LIMITS = {
    'QB': 1,
    'RB': 2,
    'WR': 2,
    'TE': 1,
    'FLEX': 1,
    'DST': 1,
    'K': 1,
    'BENCH': 6,
}
TOTAL_ROUNDS = sum(ROSTER_LIMITS.values())  # 15 rounds total

# ADJUSTED VBD BASELINES (Fixed QB/TE Overvaluation)
VBD_BASELINES = {
    'QB': 12,  # Streamer depth lowers top QB relative value
    'RB': 28,  # Expanded starter/flex depth
    'WR': 36,  # Expanded starter/flex depth
    'TE': 16,  # Streamability lowers top TE value
    'DST': 12,
    'K': 12,
}

POSITION_COLORS = {
    'QB': '#7A1C1C',   # Deep Red
    'RB': '#1E3A8A',   # Deep Navy / Blue
    'WR': '#14532D',   # Dark Forest Green
    'TE': '#7C2D12',   # Warm Orange/Brown
    'DST': '#4C1D95',  # Deep Purple
    'K': '#475569',    # Slate Gray
    'FLEX': '#581C87', # Violet
}

# ---------------------------------------------------------
# HIGH-SCORING OFFENSE BOOST FACTORS (5% - 9%)
# ---------------------------------------------------------
HIGH_SCORING_TEAMS_BOOST = {
    'LAR': 1.09, 'RAMS': 1.09, 'BUF': 1.09, 'BILLS': 1.09,
    'DET': 1.09, 'LIONS': 1.09, 'CIN': 1.09, 'BENGALS': 1.09,
    'BAL': 1.09, 'RAVENS': 1.09, 'DAL': 1.09, 'COWBOYS': 1.09,

    'CHI': 1.07, 'BEARS': 1.07, 'GB': 1.07,  'PACKERS': 1.07,
    'SF': 1.07,  '49ERS': 1.07, 'KC': 1.07,  'CHIEFS': 1.07,
    'PHI': 1.07, 'EAGLES': 1.07,

    'SEA': 1.05, 'SEAHAWKS': 1.05, 'LAC': 1.05, 'CHARGERS': 1.05,
    'NE': 1.05,  'PATRIOTS': 1.05, 'JAX': 1.05, 'JAGUARS': 1.05,
    'IND': 1.05, 'COLTS': 1.05,

    'MIA': 0.95, 'DOLPHINS': 0.95, 'NYJ': 0.95, 'JETS': 0.95,
    'NO': 0.95,  'SAINTS': 0.95,  'CAR': 0.95, 'PANTHERS': 0.95,
    'ARI': 0.95, 'CARDINALS': 0.95,
}

# ---------------------------------------------------------
# 2. LOAD & MERGE DATA FILES (WITH DYNAMIC ROOKIE MATH)
# ---------------------------------------------------------
@st.cache_data
def load_and_blend_data(r_mult):
    base_dir = os.path.dirname(__file__)

    def player_name_key(names):
        return (
            names.astype(str)
            .str.replace(r'\s+(?:Jr\.?|Sr\.?|I{1,3}|IV|V)$', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
            .str.casefold()
        )

    # 1. Primary Consensus Projections
    df_main = pd.read_csv(os.path.join(base_dir, "projections.csv"))
    df_main.columns = [c.replace('"', '').strip() for c in df_main.columns]

    # 2. 2026 ADP Source Data
    df_adp = pd.read_csv(os.path.join(base_dir, "ADP_2026.csv"))
    df_adp.columns = [c.replace('"', '').strip() for c in df_adp.columns]

    # 3. Load 2025 historical stats from Pro-Football-Reference data.
    df_pfr = pd.read_csv(os.path.join(base_dir, "history_2025_PFR.csv"))
    df_pfr['Player'] = df_pfr['Player'].str.replace(r'[*+]+$', '', regex=True).str.strip()
    df_pfr['G'] = pd.to_numeric(df_pfr['G'], errors='coerce')
    df_pfr['FantPt'] = pd.to_numeric(df_pfr['FantPt'], errors='coerce')
    df_pfr['Rec'] = pd.to_numeric(df_pfr['Rec'], errors='coerce')
    df_main['_PlayerKey'] = player_name_key(df_main['PLAYER NAME'])
    df_pfr['_PlayerKey'] = player_name_key(df_pfr['Player'])

    # Clean position mapping
    df_main['CleanPos'] = df_main['POS'].astype(str).str.extract(r'(QB|RB|WR|TE|K|DST|DEF)')
    df_main['CleanPos'] = df_main['CleanPos'].replace({'DEF': 'DST'})

    # Load 2026 fantasy-point projections for QB, RB, WR, and TE.
    projection_frames = [
        pd.read_csv(os.path.join(base_dir, "QB_proj_2026.csv")),
        pd.read_csv(os.path.join(base_dir, "FLX_proj_2026.csv")),
    ]
    df_proj = pd.concat(projection_frames, ignore_index=True)
    df_proj['Player'] = df_proj['Player'].astype(str).str.strip()
    df_proj['FPTS'] = pd.to_numeric(df_proj['FPTS'], errors='coerce')
    df_proj = df_proj[
        df_proj['Player'].ne('') & df_proj['FPTS'].notna()
    ].drop_duplicates('Player')

    # Merge ADP data into main dataset
    df = df_main.merge(
        df_adp[['Player', 'ADP', 'Bye']],
        left_on='PLAYER NAME',
        right_on='Player',
        how='left'
    )
    df['ADP'] = df['ADP'].fillna(df['AVG.'] if 'AVG.' in df.columns else 999.0)
    df = df.merge(
        df_proj[['Player', 'FPTS']],
        left_on='PLAYER NAME',
        right_on='Player',
        how='left',
    ).rename(columns={'FPTS': 'Projected_FPTS_2026'})

    # Merge 2025 half-PPR fantasy points and games played from PFR.
    df = df.merge(
        df_pfr[['_PlayerKey', 'Player', 'FantPt', 'Rec', 'G']].rename(
            columns={
                'FantPt': 'FANTASYPTS_BASE',
                'Rec': 'Receptions_2025',
                'G': 'Games_Played_2025',
            }
        ),
        left_on='_PlayerKey',
        right_on='_PlayerKey',
        how='left',
    )

    df['PROJ_PTS'] = 300 - (df['ADP'] * 1.2)
    df['FANTASYPTS'] = (
        df['FANTASYPTS_BASE'] + (0.5 * df['Receptions_2025'])
    )
    projected_positions = ['QB', 'RB', 'WR', 'TE']
    projection_mask = (
        df['CleanPos'].isin(projected_positions)
        & df['Projected_FPTS_2026'].notna()
    )
    df.loc[projection_mask, 'PROJ_PTS'] = df.loc[
        projection_mask, 'Projected_FPTS_2026'
    ]

    # Normalize each player's 2025 production to a 17-game season.
    df['FANTASYPTS_2025_17G'] = np.where(
        df['Games_Played_2025'].gt(0),
        df['FANTASYPTS'] / df['Games_Played_2025'] * 17,
        np.nan,
    )

    # Detect rookies and apply the configurable rookie upside boost.
    df['IsRookie'] = df['ROOKIE'].fillna(False).astype(bool) if 'ROOKIE' in df.columns else False
    df['Adjusted_PROJ_PTS'] = df['PROJ_PTS']
    df.loc[df['IsRookie'], 'Adjusted_PROJ_PTS'] *= r_mult

    # Blend normalized 2025 production with the 2026 projection.
    def calculate_xpts(row):
        proj = row['Adjusted_PROJ_PTS']

        # Rookies have no 2025 production to blend in.
        if row['IsRookie']:
            return proj

        pts_2025 = row['FANTASYPTS_2025_17G']
        if pd.isna(pts_2025):
            return proj
        if row['CleanPos'] == 'RB':
            return (0.15 * pts_2025) + (0.85 * proj)
        return (0.20 * pts_2025) + (0.80 * proj)

    df['xPTS'] = df.apply(calculate_xpts, axis=1)

    # Calculate Replacement Values (VBD)
    replacement_pts = {}
    for pos, rank_cutoff in VBD_BASELINES.items():
        pos_players = df[df['CleanPos'] == pos].sort_values(by='xPTS', ascending=False)
        if len(pos_players) >= rank_cutoff:
            replacement_pts[pos] = pos_players.iloc[rank_cutoff - 1]['xPTS']
        elif not pos_players.empty:
            replacement_pts[pos] = pos_players.iloc[-1]['xPTS']
        else:
            replacement_pts[pos] = 0.0

    df['VBD'] = df.apply(lambda r: r['xPTS'] - replacement_pts.get(r['CleanPos'], 0.0), axis=1)
    return df

data = load_and_blend_data(rookie_mult)

# Initialize Session States
if 'draft_history' not in st.session_state:
    st.session_state.draft_history = []
if 'my_team' not in st.session_state:
    st.session_state.my_team = []
if 'dismissed_recommendations' not in st.session_state:
    st.session_state.dismissed_recommendations = set()
if 'espn_sync_error' not in st.session_state:
    st.session_state.espn_sync_error = None

if st.sidebar.button("Restore Removed Recommendations"):
    st.session_state.dismissed_recommendations.clear()
    st.rerun()

def sync_my_team():
    st.session_state.my_team = [
        {'Name': pick['player_name'], 'Position': pick['pos']}
        for pick in st.session_state.draft_history
        if pick['team_num'] == my_pick_pos
    ]

# ---------------------------------------------------------
# 3. HELPER MATH FOR SNAKE DRAFTING
# ---------------------------------------------------------
def get_team_on_clock(pick_num, total_teams):
    rnd = ((pick_num - 1) // total_teams) + 1
    pick_in_rnd = ((pick_num - 1) % total_teams) + 1
    return pick_in_rnd if rnd % 2 != 0 else (total_teams - pick_in_rnd + 1)

def clean_player_name(name):
    if not isinstance(name, str):
        return ""
    normalized = re.sub(r"[^\w\s]", "", name)
    normalized = re.sub(
        r"\b(Jr|Sr|II|III|IV|V)\b", "", normalized, flags=re.IGNORECASE
    )
    return " ".join(normalized.split()).casefold()

def fetch_espn_draft_picks(league_id, season, espn_s2, swid):
    url = (
        "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/"
        f"seasons/{int(season)}/segments/0/leagues/{league_id}"
    )
    try:
        response = requests.get(
            url,
            params={"view": "mDraftDetail"},
            cookies={"espn_s2": espn_s2, "SWID": swid},
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
        return payload, None
    except (requests.RequestException, ValueError, TypeError) as error:
        return [], str(error)

def sync_espn_draft(league_id, season, espn_s2, swid, player_data):
    payload, error = fetch_espn_draft_picks(league_id, season, espn_s2, swid)
    if error:
        return False, 0, error

    player_lookup = {
        clean_player_name(row['PLAYER NAME']): row
        for _, row in player_data.iterrows()
    }
    espn_players = {
        str(player.get('id')): player
        for player in payload.get('players', [])
        if player.get('id') is not None
    }
    picks = payload.get('draftDetail', {}).get('picks', [])
    synced_picks = []
    position_map = {1: 'QB', 2: 'RB', 3: 'WR', 4: 'TE', 5: 'K', 16: 'DST'}
    for index, pick in enumerate(picks, 1):
        player_info = pick.get('playerPoolEntry', {}).get('player', {})
        if not player_info and pick.get('playerId') is not None:
            player_info = espn_players.get(str(pick['playerId']), {})
        player_name = player_info.get('fullName', '').strip()
        player_row = player_lookup.get(clean_player_name(player_name))
        if player_row is None or not player_name:
            continue

        pick_num = int(
            pick.get('overallPickNumber') or pick.get('pickNumber') or index
        )
        team_id = pick.get('teamId')
        team_num = (
            int(team_id)
            if team_id is not None
            else get_team_on_clock(pick_num, num_teams)
        )
        if team_num < 1 or team_num > num_teams:
            team_num = get_team_on_clock(pick_num, num_teams)
        synced_picks.append({
            'pick_num': pick_num,
            'round': int(
                pick.get('roundId') or ((pick_num - 1) // num_teams) + 1
            ),
            'team_num': team_num,
            'player_name': player_row['PLAYER NAME'],
            'pos': player_row['CleanPos'] or position_map.get(
                player_info.get('defaultPositionId'), ''
            ),
            'source': 'espn',
        })

    existing_manual = {
        pick.get('pick_num'): pick
        for pick in st.session_state.draft_history
        if pick.get('source') != 'espn'
    }
    merged = existing_manual.copy()
    merged.update({pick['pick_num']: pick for pick in synced_picks})
    new_history = [merged[pick_num] for pick_num in sorted(merged)]
    changed = new_history != st.session_state.draft_history
    if changed:
        st.session_state.draft_history = new_history
        sync_my_team()
    return changed, len(synced_picks), None

if espn_auto_sync and espn_league_id and espn_s2 and espn_swid:
    _, espn_pick_count, st.session_state.espn_sync_error = sync_espn_draft(
        espn_league_id, espn_season, espn_s2, espn_swid, data
    )
elif espn_league_id or espn_s2 or espn_swid:
    st.session_state.espn_sync_error = "Enter the league ID and both ESPN cookies to connect."
else:
    st.session_state.espn_sync_error = None

if st.session_state.espn_sync_error:
    st.sidebar.warning(st.session_state.espn_sync_error)
elif espn_auto_sync and espn_league_id:
    if espn_pick_count:
        st.sidebar.success(f"ESPN connected: {espn_pick_count} picks synced")
    else:
        st.sidebar.warning(
            "ESPN connected, but no picks were returned. Practice/mock drafts "
            "may not publish picks through the league draft API."
        )

total_picks_made = len(st.session_state.draft_history)
current_pick_num = total_picks_made + 1
current_round = ((current_pick_num - 1) // num_teams) + 1
current_team = get_team_on_clock(current_pick_num, num_teams)

def picks_until_next_turn(current_pick, user_slot, total_teams, total_rounds):
    for future_pick in range(current_pick + 1, (total_teams * total_rounds) + 1):
        if get_team_on_clock(future_pick, total_teams) == user_slot:
            return future_pick - current_pick
    return 0

picks_to_wait = picks_until_next_turn(
    current_pick_num, my_pick_pos, num_teams, TOTAL_ROUNDS
)

# Exclude drafted players
drafted_names = [p['player_name'] for p in st.session_state.draft_history]
available = data[~data['PLAYER NAME'].isin(drafted_names)].copy()

# ---------------------------------------------------------
# 4. RECOMMENDATION ENGINE (VBD + NEED MULTIPLIER + CLIFF)
# ---------------------------------------------------------
def search_player_news(player_name):
    try:
        # Keep the optional news integration from blocking app startup when
        # its native lxml dependency is unavailable or blocked by policy.
        ddgs_module = None
        for module_name in ('ddgs', 'duckduckgo_search'):
            try:
                ddgs_module = importlib.import_module(module_name)
                break
            except ImportError:
                continue
        if ddgs_module is None:
            return (
                "Web news search is unavailable because the DDGS package is "
                "not installed. Install the packages in requirements.txt."
            )
        DDGS = ddgs_module.DDGS

        with DDGS() as ddgs:
            results = list(
                ddgs.news(f"{player_name} NFL fantasy injury news", max_results=2)
            )
        if not results:
            return "No breaking news reported in the last 24 hours."
        return "\n\n".join(
            [
                f"[{r.get('source', 'Web')}] {r.get('title')}: {r.get('body')}"
                for r in results
            ]
        )
    except Exception as error:
        return f"Web news search failed: {error}"

def calculate_roster_fill(my_roster):
    counts = {
        'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0,
        'DST': 0, 'K': 0, 'FLEX': 0, 'BENCH': 0,
    }
    for p in my_roster:
        pos = p['Position']
        if counts.get(pos, 0) < ROSTER_LIMITS.get(pos, 0):
            counts[pos] += 1
        elif pos in ['RB', 'WR', 'TE'] and counts['FLEX'] < ROSTER_LIMITS['FLEX']:
            counts['FLEX'] += 1
        else:
            counts['BENCH'] += 1
    return counts

def get_roster_need_multiplier(pos, filled):
    starters_needed = ROSTER_LIMITS.get(pos, 0)
    current_filled = filled.get(pos, 0)

    if pos in ['QB', 'TE', 'K', 'DST']:
        if current_filled < starters_needed:
            return 1.05 if pos == 'TE' else 1.15
        return 0.40

    if current_filled < starters_needed:
        return 1.20
    elif filled.get('FLEX', 0) < ROSTER_LIMITS['FLEX']:
        return 1.05
    return 0.75

def calculate_positional_cliffs(df_avail):
    cliffs = {}
    for pos in ['QB', 'RB', 'WR', 'TE', 'DST', 'K']:
        pos_df = df_avail[df_avail['CleanPos'] == pos].sort_values(by='VBD', ascending=False)
        if len(pos_df) >= 3:
            drop_off = pos_df.iloc[0]['VBD'] - pos_df.iloc[2]['VBD']
            cliffs[pos] = max(0.0, drop_off)
        else:
            cliffs[pos] = 0.0
    return cliffs

def calculate_top_5_recommendations(
    df,
    my_roster,
    current_pick,
    wait_time,
    current_round,
    position_filter='ALL',
    excluded_names=None,
):
    filled = calculate_roster_fill(my_roster)
    
    df_filtered = df[
        df['CleanPos'].isin(['QB', 'RB', 'WR', 'TE', 'DST', 'K'])
    ].copy()
    df_filtered = df_filtered[
        ~df_filtered['PLAYER NAME'].isin(UNAVAILABLE_PLAYERS)
    ]
    team_labels = df_filtered['TEAM'].fillna('').astype(str).str.strip().str.upper()
    df_filtered = df_filtered[
        ~team_labels.isin(['', 'FA', 'UFA', 'FREE AGENT'])
    ]
    if filled.get('TE', 0) >= ROSTER_LIMITS['TE']:
        df_filtered = df_filtered[df_filtered['CleanPos'] != 'TE']

    if position_filter == 'FLEX (WR/RB/TE)':
        df_filtered = df_filtered[df_filtered['CleanPos'].isin(['RB', 'WR', 'TE'])]
    elif position_filter != 'ALL':
        df_filtered = df_filtered[df_filtered['CleanPos'] == position_filter]

    if excluded_names:
        df_filtered = df_filtered[
            ~df_filtered['PLAYER NAME'].isin(excluded_names)
        ]

    # Delay K/DST until round 10 and hide them after the starting slot is filled.
    unavailable_positions = {
        pos for pos in ['K', 'DST']
        if current_round < 10 or filled.get(pos, 0) >= 1
    }
    df_filtered = df_filtered[~df_filtered['CleanPos'].isin(unavailable_positions)]

    if df_filtered.empty:
        return pd.DataFrame()

    cliffs = calculate_positional_cliffs(df[df['CleanPos'].isin(['QB', 'RB', 'WR', 'TE', 'DST', 'K'])])

    adjusted_scores = []
    for _, row in df_filtered.iterrows():
        pos = row['CleanPos']
        team_str = str(row.get('TEAM', '')).strip().upper()

        base_vbd = row['VBD']
        boost_multiplier = HIGH_SCORING_TEAMS_BOOST.get(team_str, 1.0)
        need_multiplier = get_roster_need_multiplier(pos, filled)
        cliff_multiplier = 0.05 if pos == 'TE' else 0.40
        cliff_bonus = cliffs.get(pos, 0.0) * cliff_multiplier

        score = (base_vbd * boost_multiplier * need_multiplier) + cliff_bonus

        if pos == 'QB':
            score *= 0.90

        if team_str in ['FA', 'UFA', 'FREE AGENT']:
            score *= 0.5

        adjusted_scores.append(score)

    df_filtered['Adjusted_Score'] = adjusted_scores
    candidates = (
        df_filtered.sort_values(by='Adjusted_Score', ascending=False)
        .head(5)
        .copy()
    )

    next_turn_pick = current_pick + wait_time
    snipe_probs, snipe_labels = [], []
    for _, row in candidates.iterrows():
        adp = row['ADP']
        # Dynamic Standard Deviation: tighter variance in early rounds, wider variance late.
        adp_std_dev = max(3.0, adp * 0.15)
        prob = norm.cdf(next_turn_pick, loc=adp, scale=adp_std_dev) * 100.0
        snipe_probs.append(prob)

        if wait_time == 0:
            snipe_labels.append("🎯 ON CLOCK")
        elif prob >= 65.0:
            snipe_labels.append(f"🚨 HIGH RISK ({prob:.0f}%)")
        elif prob >= 35.0:
            snipe_labels.append(f"⚠️ MOD RISK ({prob:.0f}%)")
        else:
            snipe_labels.append(f"🟢 SAFE ({prob:.0f}%)")

    candidates['SnipeProb'] = snipe_probs
    candidates['SnipeLabel'] = snipe_labels
    return candidates

def style_draft_board(val):
    if not isinstance(val, str) or val == "—":
        return ""
    for pos, color in POSITION_COLORS.items():
        if val.startswith(f"[{pos}]"):
            return (
                f"background-color: {color}; color: white; font-weight: bold;"
            )
    return ""


def style_risk_badge(label):
    if not isinstance(label, str):
        return "**—**"

    text = label.upper()
    if "HIGH RISK" in text:
        bg_color, text_color = "#dc2626", "#ffffff"
    elif "MOD RISK" in text:
        bg_color, text_color = "#facc15", "#1f2937"
    elif "SAFE" in text or "ON CLOCK" in text:
        bg_color, text_color = "#22c55e", "#ffffff"
    else:
        return f"**{label}**"

    return (
        '<span style="display:inline-block; padding:0.25rem 0.55rem; '
        'border-radius:999px; font-size:0.78rem; font-weight:700; '
        f'background-color:{bg_color}; color:{text_color}; '
        'line-height:1.2; white-space:nowrap;">'
        f'{label}</span>'
    )

# ---------------------------------------------------------
# 5. STREAMLIT LAYOUT DEFINITION (COLUMNS)
# ---------------------------------------------------------
col_main, col_side = st.columns([1.3, 0.7])

# --- LEFT / MAIN COLUMN ---
with col_main:
    is_my_turn = current_team == my_pick_pos
    st.subheader(
        f"📍 Pick #{current_pick_num} (Round {current_round}) — Team {current_team}'s Turn"
    )
    if is_my_turn:
        st.error("🚨 YOU ARE ON THE CLOCK!")
    else:
        st.info(f"⏳ Picks until your next turn: **{picks_to_wait}**")

    head_col, filter_col = st.columns([0.65, 0.35])
    with head_col:
        st.header("🎯 Top 5 Suggested Picks")
    with filter_col:
        selected_pos_filter = st.selectbox(
            "Filter by Position:",
            options=['ALL', 'FLEX (WR/RB/TE)', 'RB', 'WR', 'TE', 'QB', 'DST', 'K'],
            index=0,
            key="pos_filter_select"
        )

    top_5 = calculate_top_5_recommendations(
        available, 
        st.session_state.my_team, 
        current_pick_num, 
        picks_to_wait, 
        current_round,
        position_filter=selected_pos_filter,
        excluded_names=st.session_state.dismissed_recommendations,
    )

    if top_5.empty:
        st.write("*No available players match the selected position filter.*")
    else:
        st.markdown(
            """
            <style>
            [class*="st-key-recommendation_tile"] {
                padding: 0.65rem 0.8rem;
            }
            [class*="st-key-recommendation_tile"] [data-testid="stVerticalBlock"] {
                gap: 0.35rem;
            }
            [class*="st-key-recommendation_tile"] .recommendation-player-name {
                font-size: 125%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        for rank, (_, row) in enumerate(top_5.iterrows(), 1):
            p_name = row['PLAYER NAME']
            bye_val = row.get('Bye', '—')

            with st.container(
                border=True,
                key=f"recommendation_tile_{rank}",
            ):
                rank_col, player_col, team_col, bye_col, risk_col, aim_col, draft_col = st.columns(
                    [0.06, 0.27, 0.14, 0.10, 0.18, 0.11, 0.14]
                )
                with rank_col:
                    st.markdown(f"**#{rank}**")
                with player_col:
                    st.markdown(
                        f'<span class="recommendation-player-name"><strong>{p_name}</strong></span>',
                        unsafe_allow_html=True,
                    )
                with team_col:
                    st.markdown(f"`{row['TEAM']} - {row['CleanPos']}`")
                with bye_col:
                    st.markdown(f"Bye: **{bye_val}**")
                with risk_col:
                    st.markdown(style_risk_badge(row['SnipeLabel']), unsafe_allow_html=True)
                with aim_col:
                    st.markdown(f"AIM: **{row['Adjusted_Score']:.1f}**")
                with draft_col:
                    if st.button(
                        f"Drafted by Team {current_team}",
                        key=f"draft_{p_name}",
                        use_container_width=True,
                    ):
                        st.session_state.draft_history.append({
                            'pick_num': current_pick_num,
                            'round': current_round,
                            'team_num': current_team,
                            'player_name': p_name,
                            'pos': row['CleanPos'],
                            'aim_score': row['Adjusted_Score'],
                        })
                        sync_my_team()
                        st.rerun()

                with st.expander("Details"):
                    metric_cols = st.columns(5)
                    metric_cols[0].metric("VBD", f"{row['VBD']:+.1f}")
                    metric_cols[1].metric("xPTS", f"{row['xPTS']:.1f}")
                    metric_cols[2].metric("Proj", f"{row['Adjusted_PROJ_PTS']:.1f}")
                    metric_cols[3].metric("ADP", f"{row['ADP']:.1f}")
                    metric_cols[4].metric("AIM", f"{row['Adjusted_Score']:.1f}")

                    detail_actions = st.columns(2)
                    with detail_actions[0]:
                        if st.button(
                            "Search Web News",
                            key=f"search_{p_name}",
                            use_container_width=True,
                        ):
                            with st.spinner("Searching latest news..."):
                                st.info(search_player_news(p_name))
                    with detail_actions[1]:
                        if st.button(
                            "Remove",
                            key=f"remove_{p_name}",
                            use_container_width=True,
                        ):
                            st.session_state.dismissed_recommendations.add(p_name)
                            st.rerun()

    st.header("📊 Interactive Snake Draft Board")
    if st.session_state.draft_history:
        board_df = pd.DataFrame(st.session_state.draft_history)
        board_df['display_text'] = board_df.apply(
            lambda r: (
                f"[{r['pos']}] {r['player_name']} (AIM: {r['aim_score']:.1f})"
                if pd.notna(r.get('aim_score'))
                else f"[{r['pos']}] {r['player_name']}"
            ),
            axis=1,
        )
        board_pivot = board_df.pivot(
            index='round', columns='team_num', values='display_text'
        ).fillna("—")

        styled_board = board_pivot.style.map(style_draft_board)
        st.dataframe(styled_board, use_container_width=True)
    else:
        st.write("*Draft has not started yet.*")

# --- RIGHT / SIDE COLUMN ---
with col_side:
    st.header("⚡ Quick Select: Top 10 ADP Available")
    top_10_adp = available[
        ~available['PLAYER NAME'].isin(UNAVAILABLE_PLAYERS)
    ].sort_values(by='ADP', ascending=True).head(10)

    if not top_10_adp.empty:
        for idx, (_, t10_row) in enumerate(top_10_adp.iterrows(), 1):
            t10_name = t10_row['PLAYER NAME']
            t10_pos = t10_row['CleanPos']
            t10_team = t10_row['TEAM']
            t10_adp_val = t10_row['ADP']

            btn_label = f"#{idx} [{t10_pos}] {t10_name} ({t10_team}) - ADP: {t10_adp_val:.1f}"

            if st.button(
                btn_label,
                key=f"quick_draft_{t10_name}",
                use_container_width=True,
            ):
                st.session_state.draft_history.append({
                    'pick_num': current_pick_num,
                    'round': current_round,
                    'team_num': current_team,
                    'player_name': t10_name,
                    'pos': t10_pos,
                })
                sync_my_team()
                st.rerun()
    else:
        st.write("*No more players available.*")

    st.divider()

    st.header("🔍 Manual Dropdown Log")
    selected_player = st.selectbox(
        "Select player drafted:",
        options=[""] + list(available['PLAYER NAME'].sort_values()),
        key="manual_draft_select",
    )
    if st.button("Confirm Pick") and selected_player:
        pos = available[available['PLAYER NAME'] == selected_player][
            'CleanPos'
        ].values[0]
        st.session_state.draft_history.append({
            'pick_num': current_pick_num,
            'round': current_round,
            'team_num': current_team,
            'player_name': selected_player,
            'pos': pos,
        })
        sync_my_team()
        st.rerun()

    st.divider()

    st.header("🛠️ Draft Corrections")
    if st.session_state.draft_history:
        with st.expander("✏️ Correct / Delete a Specific Pick"):
            pick_options = [
                f"Pick #{p['pick_num']} (R{p['round']}) - Team {p['team_num']}: {p['player_name']}"
                for p in st.session_state.draft_history
            ]

            selected_to_remove = st.selectbox(
                "Select pick to remove:", options=pick_options
            )

            if st.button("❌ Remove Selected Pick"):
                idx_to_remove = pick_options.index(selected_to_remove)
                removed_pick = st.session_state.draft_history.pop(idx_to_remove)

                for i, pick in enumerate(st.session_state.draft_history):
                    pick['pick_num'] = i + 1
                    pick['round'] = ((i) // num_teams) + 1
                    pick['team_num'] = get_team_on_clock(
                        pick['pick_num'], num_teams
                    )

                sync_my_team()
                st.success(f"Removed {removed_pick['player_name']}!")
                st.rerun()

        if st.button("Undo Very Last Pick"):
            st.session_state.draft_history.pop()
            sync_my_team()
            st.rerun()
    else:
        st.write("*No picks to correct yet.*")

    st.divider()

    st.header("📋 My Roster Tracker")
    roster_counts = calculate_roster_fill(st.session_state.my_team)
    for slot, limit in ROSTER_LIMITS.items():
        st.write(f"**{slot}:** {roster_counts[slot]} / {limit}")

    st.subheader("Roster Players:")
    if st.session_state.my_team:
        for p in st.session_state.my_team:
            st.write(f"• **{p['Position']}:** {p['Name']}")
    else:
        st.write("*No players drafted yet.*")

if espn_auto_sync and espn_league_id and espn_s2 and espn_swid:
    time.sleep(espn_sync_seconds)
    st.rerun()
