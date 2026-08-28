import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st

# Preferred import with fallback to legacy package name
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

st.set_page_config(page_title="Snake Master Draft Assistant", layout="wide")
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

st.sidebar.header("⚖️ Value Adjustments")

# Dynamic Sliders for Rookies & Returning Injured Players
rookie_boost_pct = st.sidebar.slider(
    "Rookie Upside Boost (%)",
    min_value=0,
    max_value=25,
    value=5,
    step=1,
    help="Increases 2026 projections for rookies to account for unmapped ceiling."
)

injury_bounceback_pct = st.sidebar.slider(
    "Injury Recovery Multiplier (%)",
    min_value=0,
    max_value=20,
    value=5,
    step=1,
    help="Applies a moderated boost to 2026 projections for players who missed significant time in 2025."
)

# Multiplier Conversions
rookie_mult = 1.0 + (rookie_boost_pct / 100.0)
injury_mult = 1.0 + (injury_bounceback_pct / 100.0)

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
# 2. LOAD & MERGE DATA FILES (WITH DYNAMIC ROOKIE/INJURY MATH)
# ---------------------------------------------------------
@st.cache_data
def load_and_blend_data(r_mult, i_mult):
    base_dir = os.path.dirname(__file__)

    # 1. Primary Consensus Projections
    df_main = pd.read_csv(os.path.join(base_dir, "projections.csv"))
    df_main.columns = [c.replace('"', '').strip() for c in df_main.columns]

    # 2. 2026 ADP Source Data
    df_adp = pd.read_csv(os.path.join(base_dir, "ADP_2026.csv"))
    df_adp.columns = [c.replace('"', '').strip() for c in df_adp.columns]

    # 3. Load 2025 historical stats from Pro-Football-Reference data.
    df_pfr = pd.read_csv(os.path.join(base_dir, "history_2025_PFR.csv"))
    df_pfr['Player'] = df_pfr['Player'].str.replace(r'[*+]+$', '', regex=True).str.strip()
    df_pfr['GS'] = pd.to_numeric(df_pfr['GS'], errors='coerce')
    df_pfr['FantPt'] = pd.to_numeric(df_pfr['FantPt'], errors='coerce')

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

    # Merge 2025 half-PPR fantasy points and games started from PFR.
    df = df.merge(
        df_pfr[['Player', 'FantPt', 'GS']].rename(
            columns={'FantPt': 'FANTASYPTS', 'GS': 'Games_Started_2025'}
        ),
        left_on='PLAYER NAME',
        right_on='Player',
        how='left',
    )

    df['PROJ_PTS'] = 300 - (df['ADP'] * 1.2)
    projected_positions = ['QB', 'RB', 'WR', 'TE']
    projection_mask = (
        df['CleanPos'].isin(projected_positions)
        & df['Projected_FPTS_2026'].notna()
    )
    df.loc[projection_mask, 'PROJ_PTS'] = df.loc[
        projection_mask, 'Projected_FPTS_2026'
    ]

    # Detect Rookies and Returning Injured Players
    df['IsRookie'] = df['ROOKIE'].fillna(False).astype(bool) if 'ROOKIE' in df.columns else False
    df['InjuredLastYear'] = (
        df['Games_Started_2025'].lt(17) & ~df['IsRookie']
    )
    df['Adjusted_PROJ_PTS'] = df['PROJ_PTS']
    df.loc[df['IsRookie'], 'Adjusted_PROJ_PTS'] *= r_mult
    effective_i_mult = 1.0 + ((i_mult - 1.0) * 0.5)
    df.loc[df['InjuredLastYear'], 'Adjusted_PROJ_PTS'] *= effective_i_mult

    # Dynamic xPTS calculation incorporating sidebar multipliers
    def calculate_xpts(row):
        proj = row['Adjusted_PROJ_PTS']
        
        # 1. Rookies: Apply dynamic rookie upside multiplier directly to 2026 projections
        if row['IsRookie']:
            return proj
            
        # 2. Injured Last Year: Apply dynamic recovery multiplier to 2026 projections
        if row['InjuredLastYear']:
            return proj
            
        # 3. Healthy Veterans: Position-specific historical weighting
        pts_2025 = row['FANTASYPTS']
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

data = load_and_blend_data(rookie_mult, injury_mult)

# Initialize Session States
if 'draft_history' not in st.session_state:
    st.session_state.draft_history = []
if 'my_team' not in st.session_state:
    st.session_state.my_team = []
if 'dismissed_recommendations' not in st.session_state:
    st.session_state.dismissed_recommendations = set()

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
total_picks_made = len(st.session_state.draft_history)
current_pick_num = total_picks_made + 1
current_round = ((current_pick_num - 1) // num_teams) + 1

def get_team_on_clock(pick_num, total_teams):
    rnd = ((pick_num - 1) // total_teams) + 1
    pick_in_rnd = ((pick_num - 1) % total_teams) + 1
    return pick_in_rnd if rnd % 2 != 0 else (total_teams - pick_in_rnd + 1)

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
    except Exception:
        return "Web news search unavailable."

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
            </style>
            """,
            unsafe_allow_html=True,
        )
        for rank, (_, row) in enumerate(top_5.iterrows(), 1):
            p_name = row['PLAYER NAME']
            bye_val = row.get('Bye', '—')

            # Create visual status tags for Rookies / Injured Players
            tags = []
            if row['IsRookie']:
                tags.append("`[ROOKIE]`")
            if row['InjuredLastYear']:
                tags.append("`[RECOVERING]`")
            tag_str = " " + " ".join(tags) if tags else ""

            with st.container(
                border=True,
                key=f"recommendation_tile_{rank}",
            ):
                h1, h2 = st.columns([0.7, 0.3])
                with h1:
                    st.markdown(
                        f"**#{rank} {p_name}** `[{row['CleanPos']} - {row['TEAM']}]`{tag_str} (Bye: {bye_val})"
                    )
                with h2:
                    if "🚨" in row['SnipeLabel']:
                        st.error(row['SnipeLabel'], icon=None)
                    elif "⚠️" in row['SnipeLabel']:
                        st.warning(row['SnipeLabel'], icon=None)
                    else:
                        st.success(row['SnipeLabel'], icon=None)

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("VBD", f"{row['VBD']:+.1f}")
                m2.metric("xPTS", f"{row['xPTS']:.1f}")
                m3.metric("Proj", f"{row['Adjusted_PROJ_PTS']:.1f}")
                m4.metric("2026 ADP", f"{row['ADP']:.1f}")
                m5.metric("AIM Score", f"{row['Adjusted_Score']:.1f}")

                a1, a2 = st.columns([0.7, 0.3])
                with a1:
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
                with a2:
                    if st.button(
                        "Remove",
                        key=f"remove_{p_name}",
                        use_container_width=True,
                    ):
                        st.session_state.dismissed_recommendations.add(p_name)
                        st.rerun()
                    with st.expander("🔍 News"):
                        if st.button(
                            f"Search Web News",
                            key=f"search_{p_name}",
                            use_container_width=True,
                        ):
                            with st.spinner("Searching latest news..."):
                                st.info(search_player_news(p_name))

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
    top_10_adp = available.sort_values(by='ADP', ascending=True).head(10)

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