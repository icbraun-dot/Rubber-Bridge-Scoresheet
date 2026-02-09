
import streamlit as st
import sqlite3
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path

st.set_page_config(page_title="Rubber Bridge — Score + Stats", layout="wide")

# ============================================================
# Persistent storage (SQLite)
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "rubber_bridge.sqlite"

def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS players (
        player_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS rubbers (
        rubber_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        north_player_id INTEGER NOT NULL,
        south_player_id INTEGER NOT NULL,
        east_player_id  INTEGER NOT NULL,
        west_player_id  INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'active', -- active|complete
        winner_side TEXT,                      -- NS|EW
        ns_above INTEGER NOT NULL DEFAULT 0,
        ns_below INTEGER NOT NULL DEFAULT 0,
        ew_above INTEGER NOT NULL DEFAULT 0,
        ew_below INTEGER NOT NULL DEFAULT 0,
        ns_games INTEGER NOT NULL DEFAULT 0,
        ew_games INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY(north_player_id) REFERENCES players(player_id),
        FOREIGN KEY(south_player_id) REFERENCES players(player_id),
        FOREIGN KEY(east_player_id)  REFERENCES players(player_id),
        FOREIGN KEY(west_player_id)  REFERENCES players(player_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS deals (
        deal_id INTEGER PRIMARY KEY AUTOINCREMENT,
        rubber_id INTEGER NOT NULL,
        deal_no INTEGER NOT NULL,
        side TEXT NOT NULL,               -- declarer side: NS|EW
        declarer_player_id INTEGER NOT NULL,
        level INTEGER NOT NULL,
        strain TEXT NOT NULL,             -- C D H S NT
        doubled INTEGER NOT NULL,          -- 0/1/2
        tricks_result INTEGER NOT NULL,    -- +over, 0 exact, negative = down
        vul INTEGER NOT NULL,              -- 0/1 vulnerability of declarer side
        made INTEGER NOT NULL,             -- 0/1
        below INTEGER NOT NULL DEFAULT 0,
        above INTEGER NOT NULL DEFAULT 0,
        defense_above INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(rubber_id) REFERENCES rubbers(rubber_id) ON DELETE CASCADE,
        FOREIGN KEY(declarer_player_id) REFERENCES players(player_id)
    );
    """)

    conn.commit()
    conn.close()

init_db()

# ============================================================
# Rubber scoring helpers (starter rules, extend as desired)
# ============================================================

TRICK_VALUES = {"C": 20, "D": 20, "H": 30, "S": 30, "NT": 30}

def contract_trick_score(level: int, strain: str, doubled: int) -> int:
    if strain == "NT":
        base = 40 + (level - 1) * 30
    else:
        base = level * TRICK_VALUES[strain]
    if doubled == 1:
        base *= 2
    elif doubled == 2:
        base *= 4
    return base

def overtrick_score(over: int, strain: str, doubled: int, vulnerable: bool) -> int:
    if over <= 0:
        return 0
    if doubled == 0:
        return over * (30 if strain == "NT" else TRICK_VALUES[strain])
    mult = 1 if doubled == 1 else 2
    return over * (200 if vulnerable else 100) * mult

def undertrick_penalty(under: int, doubled: int, vulnerable: bool) -> int:
    if under <= 0:
        return 0
    if doubled == 0:
        return under * (100 if vulnerable else 50)

    mult = 1 if doubled == 1 else 2
    if vulnerable:
        # 1st:200, 2nd+:300 each (doubled), redoubled x2
        if under == 1:
            return 200 * mult
        return (200 + (under - 1) * 300) * mult
    else:
        # 1st:100, 2nd&3rd:200 each, 4th+:300 each (doubled), redoubled x2
        if under == 1:
            return 100 * mult
        if under == 2:
            return (100 + 200) * mult
        if under == 3:
            return (100 + 200 + 200) * mult
        return (100 + 200 + 200 + (under - 3) * 300) * mult

def insult_bonus(doubled: int) -> int:
    return 50 if doubled == 1 else (100 if doubled == 2 else 0)

def slam_bonus(level: int, vulnerable: bool) -> int:
    if level == 6:
        return 750 if vulnerable else 500
    if level == 7:
        return 1500 if vulnerable else 1000
    return 0

def partscore_or_game_bonus(below_line_points: int, vulnerable: bool) -> int:
    if below_line_points >= 100:
        return 500 if vulnerable else 300
    return 50

def rubber_bonus(opponent_games: int) -> int:
    # typical rubber bonus: 700 if opp has 0 games, else 500
    return 700 if opponent_games == 0 else 500

# ============================================================
# DB helpers
# ============================================================

def get_or_create_player(name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Player name cannot be blank.")
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO players(name) VALUES (?)", (name,))
    conn.commit()
    cur.execute("SELECT player_id FROM players WHERE name = ?", (name,))
    pid = cur.fetchone()[0]
    conn.close()
    return pid

def list_players() -> pd.DataFrame:
    conn = db_conn()
    df = pd.read_sql_query("SELECT player_id, name FROM players ORDER BY name COLLATE NOCASE", conn)
    conn.close()
    return df

def create_rubber(n: int, s: int, e: int, w: int) -> int:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO rubbers(created_at, north_player_id, south_player_id, east_player_id, west_player_id, status)
        VALUES (datetime('now'), ?, ?, ?, ?, 'active')
    """, (n, s, e, w))
    rid = cur.lastrowid
    conn.commit()
    conn.close()
    return rid

def list_rubbers(status_filter: str | None = None) -> pd.DataFrame:
    conn = db_conn()
    where = ""
    params = ()
    if status_filter:
        where = "WHERE r.status = ?"
        params = (status_filter,)
    df = pd.read_sql_query(f"""
        SELECT
            r.rubber_id,
            r.created_at,
            r.status,
            r.winner_side,
            pn.name AS North, ps.name AS South, pe.name AS East, pw.name AS West,
            r.ns_above, r.ns_below, r.ew_above, r.ew_below, r.ns_games, r.ew_games
        FROM rubbers r
        JOIN players pn ON pn.player_id = r.north_player_id
        JOIN players ps ON ps.player_id = r.south_player_id
        JOIN players pe ON pe.player_id = r.east_player_id
        JOIN players pw ON pw.player_id = r.west_player_id
        {where}
        ORDER BY r.rubber_id DESC
    """, conn, params=params)
    conn.close()
    return df

def get_rubber_state(rubber_id: int) -> dict:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT ns_above, ns_below, ew_above, ew_below, ns_games, ew_games, status, winner_side
        FROM rubbers WHERE rubber_id = ?
    """, (rubber_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise ValueError("Rubber not found.")
    return {
        "ns_above": row[0], "ns_below": row[1],
        "ew_above": row[2], "ew_below": row[3],
        "ns_games": row[4], "ew_games": row[5],
        "status": row[6], "winner_side": row[7]
    }

def update_rubber_state(rubber_id: int, **kwargs):
    allowed = {"ns_above","ns_below","ew_above","ew_below","ns_games","ew_games","status","winner_side"}
    sets = []
    vals = []
    for k, v in kwargs.items():
        if k not in allowed:
            continue
        sets.append(f"{k} = ?")
        vals.append(v)
    if not sets:
        return
    vals.append(rubber_id)
    conn = db_conn()
    conn.execute(f"UPDATE rubbers SET {', '.join(sets)} WHERE rubber_id = ?", vals)
    conn.commit()
    conn.close()

def next_deal_no(rubber_id: int) -> int:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(deal_no),0) + 1 FROM deals WHERE rubber_id = ?", (rubber_id,))
    n = cur.fetchone()[0]
    conn.close()
    return int(n)

def insert_deal(rubber_id: int, record: dict):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO deals(
            rubber_id, deal_no, side, declarer_player_id,
            level, strain, doubled, tricks_result, vul, made,
            below, above, defense_above, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        rubber_id, record["deal_no"], record["side"], record["declarer_player_id"],
        record["level"], record["strain"], record["doubled"], record["tricks_result"],
        int(record["vul"]), int(record["made"]),
        record["below"], record["above"], record["defense_above"]
    ))
    conn.commit()
    conn.close()

def deals_df(rubber_id: int) -> pd.DataFrame:
    conn = db_conn()
    df = pd.read_sql_query("""
        SELECT
            d.deal_no, d.side, p.name AS declarer,
            d.level, d.strain, d.doubled, d.tricks_result,
            d.vul, d.made, d.below, d.above, d.defense_above, d.created_at
        FROM deals d
        JOIN players p ON p.player_id = d.declarer_player_id
        WHERE d.rubber_id = ?
        ORDER BY d.deal_no ASC
    """, conn, params=(rubber_id,))
    conn.close()
    return df

# ============================================================
# Rubber logic
# ============================================================

def current_vulnerability_from_games(side: str, ns_games: int, ew_games: int) -> bool:
    return (ns_games >= 1) if side == "NS" else (ew_games >= 1)

def apply_deal_to_state(state: dict, side: str, made: bool, below: int, above: int, defense_above: int) -> dict:
    ns_above, ns_below = state["ns_above"], state["ns_below"]
    ew_above, ew_below = state["ew_above"], state["ew_below"]
    ns_games, ew_games = state["ns_games"], state["ew_games"]

    if made:
        if side == "NS":
            ns_below += below
            ns_above += above
        else:
            ew_below += below
            ew_above += above
    else:
        # defenders score above-the-line
        if side == "NS":
            ew_above += defense_above
        else:
            ns_above += defense_above

    # check game completion
    if ns_below >= 100:
        ns_games += 1
        ns_below = 0
    if ew_below >= 100:
        ew_games += 1
        ew_below = 0

    # rubber completion
    status = state["status"]
    winner = state["winner_side"]
    if status == "active" and (ns_games == 2 or ew_games == 2):
        status = "complete"
        if ns_games == 2:
            winner = "NS"
            ns_above += rubber_bonus(opponent_games=ew_games)
        else:
            winner = "EW"
            ew_above += rubber_bonus(opponent_games=ns_games)

    return {
        "ns_above": ns_above, "ns_below": ns_below,
        "ew_above": ew_above, "ew_below": ew_below,
        "ns_games": ns_games, "ew_games": ew_games,
        "status": status, "winner_side": winner
    }

# ============================================================
# UI
# ============================================================

st.title("Rubber Bridge — Running Tab Across Multiple Rubbers")

# --- Sidebar: Player management + Rubber selection ---
with st.sidebar:
    st.header("Players")

    new_name = st.text_input("Add a player", placeholder="Type name and press Add")
    if st.button("Add player"):
        try:
            get_or_create_player(new_name)
            st.success("Player added.")
        except Exception as e:
            st.error(str(e))

    players = list_players()
    st.caption(f"{len(players)} players saved in database")

    st.divider()
    st.header("Rubbers")

    rb_df = list_rubbers()
    if rb_df.empty:
        st.info("No rubbers yet. Create one below.")
        rubber_id = None
    else:
        # Build a readable label for selection
        rb_df = rb_df.copy()
        rb_df["label"] = rb_df.apply(
            lambda r: f'#{int(r["rubber_id"])} • {r["status"]} • {r["created_at"]} • NS({r["North"]}/{r["South"]}) vs EW({r["East"]}/{r["West"]})',
            axis=1
        )
        rubber_id = st.selectbox("Select rubber", rb_df["rubber_id"].tolist(), format_func=lambda x: rb_df.loc[rb_df["rubber_id"]==x, "label"].iloc[0])

    st.divider()
    st.subheader("Create new rubber")

    if players.empty:
        st.warning("Add players first.")
    else:
        name_map = dict(zip(players["name"], players["player_id"]))
        names = players["name"].tolist()

        c1, c2 = st.columns(2)
        with c1:
            north = st.selectbox("North", names, index=0)
            south = st.selectbox("South", names, index=min(1, len(names)-1))
        with c2:
            east = st.selectbox("East", names, index=min(2, len(names)-1))
            west = st.selectbox("West", names, index=min(3, len(names)-1))

        if st.button("Start rubber"):
            if len({north, south, east, west}) < 4:
                st.error("Choose 4 different players.")
            else:
                rid = create_rubber(name_map[north], name_map[south], name_map[east], name_map[west])
                st.success(f"Created rubber #{rid}")
                st.rerun()

# --- Main area: show rubber, enter deals, stats ---
if rubber_id is None:
    st.stop()

state = get_rubber_state(rubber_id)
rubber_row = list_rubbers().query("rubber_id == @rubber_id").iloc[0]
ns_names = f'{rubber_row["North"]} / {rubber_row["South"]}'
ew_names = f'{rubber_row["East"]} / {rubber_row["West"]}'

top1, top2, top3 = st.columns([1.4, 1.2, 1.4])

with top1:
    st.subheader(f"Rubber #{rubber_id}")
    st.write(f"**NS:** {ns_names}")
    st.write(f"**EW:** {ew_names}")
    if state["status"] == "complete":
        st.success(f'Complete — Winner: **{state["winner_side"]}**')

with top2:
    st.subheader("Scoreboard")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("NS Above", state["ns_above"])
        st.metric("NS Below", state["ns_below"])
        st.metric("NS Games", state["ns_games"])
    with c2:
        st.metric("EW Above", state["ew_above"])
        st.metric("EW Below", state["ew_below"])
        st.metric("EW Games", state["ew_games"])

with top3:
    st.subheader("Totals")
    st.write(f"**NS total (above+current below):** {state['ns_above'] + state['ns_below']}")
    st.write(f"**EW total (above+current below):** {state['ew_above'] + state['ew_below']}")

st.divider()

left, right = st.columns([1.25, 1])

# -----------------------------
# Deal entry
# -----------------------------
with left:
    st.subheader("Enter Deal Result")
    if state["status"] == "complete":
        st.warning("This rubber is complete. Start a new rubber to enter more deals.")

    deal_no = next_deal_no(rubber_id)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        side = st.selectbox("Declarer side", ["NS", "EW"], index=0, disabled=(state["status"]=="complete"))
    with c2:
        # declarer list depends on side
        if side == "NS":
            declarer_name = st.selectbox("Declarer", [rubber_row["North"], rubber_row["South"]], disabled=(state["status"]=="complete"))
        else:
            declarer_name = st.selectbox("Declarer", [rubber_row["East"], rubber_row["West"]], disabled=(state["status"]=="complete"))
    with c3:
        level = st.selectbox("Level", [1,2,3,4,5,6,7], index=2, disabled=(state["status"]=="complete"))
    with c4:
        strain = st.selectbox("Strain", ["C","D","H","S","NT"], index=4, disabled=(state["status"]=="complete"))

    c5, c6, c7 = st.columns([1, 1, 2])
    with c5:
        dbl_label = st.selectbox("Double", ["Undoubled", "Doubled", "Redoubled"], index=0, disabled=(state["status"]=="complete"))
        doubled = {"Undoubled":0, "Doubled":1, "Redoubled":2}[dbl_label]
    with c6:
        result_mode = st.selectbox("Result", ["Made", "Down"], index=0, disabled=(state["status"]=="complete"))
    with c7:
        if result_mode == "Made":
            over = st.number_input("Overtricks", min_value=0, max_value=max(0, 7 - int(level)), value=0, step=1, disabled=(state["status"]=="complete"))
            tricks_result = int(over)
            made = True
        else:
            under = st.number_input("Undertricks", min_value=1, max_value=max(1, 6 + int(level)), value=1, step=1, disabled=(state["status"]=="complete"))
            tricks_result = -int(under)
            made = False

    vul = current_vulnerability_from_games(side, state["ns_games"], state["ew_games"])
    st.info(f"Vulnerability (auto): **{side} is {'VUL' if vul else 'NOT vul'}**")

    below = above = defense_above = 0
    if made:
        below = contract_trick_score(level, strain, doubled)
        above += overtrick_score(tricks_result, strain, doubled, vul)
        above += insult_bonus(doubled)
        above += slam_bonus(level, vul)
        above += partscore_or_game_bonus(below, vul)
    else:
        defense_above = undertrick_penalty(-tricks_result, doubled, vul)

    st.write("### Preview")
    if made:
        st.write(f"- Below the line ({side}): **{below}**")
        st.write(f"- Above the line ({side}): **{above}**")
    else:
        defenders = "EW" if side == "NS" else "NS"
        st.write(f"- Set: defenders ({defenders}) score above the line: **{defense_above}**")

    if st.button("Add Deal", disabled=(state["status"]=="complete")):
        # look up declarer player_id
        declarer_pid = get_or_create_player(declarer_name)

        deal = {
            "deal_no": deal_no,
            "side": side,
            "declarer_player_id": declarer_pid,
            "level": int(level),
            "strain": strain,
            "doubled": int(doubled),
            "tricks_result": int(tricks_result),
            "vul": bool(vul),
            "made": bool(made),
            "below": int(below),
            "above": int(above),
            "defense_above": int(defense_above),
        }

        insert_deal(rubber_id, deal)

        # apply to rubber totals
        new_state = apply_deal_to_state(state, side, made, below, above, defense_above)
        update_rubber_state(rubber_id, **new_state)
        st.rerun()

# -----------------------------
# Deal Log + rubber stats
# -----------------------------
with right:
    st.subheader("Deal Log")
    df = deals_df(rubber_id)

    if df.empty:
        st.info("No deals yet.")
    else:
        df = df.copy()
        df["contract"] = df["level"].astype(str) + df["strain"] + df["doubled"].map({0:"",1:"X",2:"XX"})
        df["result"] = df.apply(
            lambda r: ("+" + str(int(r["tricks_result"]))) if (r["made"]==1 and r["tricks_result"]>0)
            else ("=" if r["made"]==1 else str(int(r["tricks_result"]))),
            axis=1
        )
        show = df[["deal_no","side","declarer","contract","result","vul","below","above","defense_above"]]
        st.dataframe(show, use_container_width=True, height=360)

        st.subheader("Basic Rubber Stats")
        df["made_int"] = df["made"].astype(int)
        df["doubled_flag"] = (df["doubled"] > 0).astype(int)

        by_player = df.groupby("declarer").agg(
            deals=("deal_no","count"),
            made=("made_int","sum"),
            set=("made_int", lambda s: int((s==0).sum())),
            avg_overtricks=("tricks_result", lambda s: float(s[s>0].mean()) if (s>0).any() else 0.0),
            doubled_rate=("doubled_flag", "mean")
        ).reset_index().sort_values(["deals","made"], ascending=False)

        st.write("**By declarer (this rubber)**")
        st.dataframe(by_player, use_container_width=True)

st.divider()
st.subheader("Across-All-Rubbers Player Statistics")

# Across all rubbers: totals by player as declarer (starter stat)
conn = db_conn()
all_df = pd.read_sql_query("""
    SELECT
        p.name AS player,
        COUNT(*) AS deals_declared,
        SUM(d.made) AS made,
        SUM(CASE WHEN d.made = 0 THEN 1 ELSE 0 END) AS sets,
        AVG(CASE WHEN d.tricks_result > 0 AND d.made = 1 THEN d.tricks_result ELSE NULL END) AS avg_overtricks,
        AVG(CASE WHEN d.doubled > 0 THEN 1.0 ELSE 0.0 END) AS doubled_rate
    FROM deals d
    JOIN players p ON p.player_id = d.declarer_player_id
    GROUP BY p.player_id
    ORDER BY deals_declared DESC, made DESC
""", conn)
conn.close()

if all_df.empty:
    st.info("No deal history yet. Enter some deals to build stats.")
else:
    all_df = all_df.fillna({"avg_overtricks": 0})
    st.dataframe(all_df, use_container_width=True)

st.caption("Database file is saved next to this app as rubber_bridge.sqlite")

