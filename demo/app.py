import json
import os
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "output" / "results"
PLOTS_DIR = BASE_DIR / "output" / "plots"
BEHAVIOR_GROUPS_PATH = RESULTS_DIR / "user_behavior_groups.csv"
ACCOUNT_HISTORY_PATH = RESULTS_DIR / "demo_account_histories.json"
DATASET_DIR = BASE_DIR / "dataset"
LISTENING_PATH = DATASET_DIR / "userid-timestamp-artid-artname-traid-traname.tsv"

LISTENING_COLUMNS = [
    "userid",
    "timestamp",
    "artist_id",
    "artist_name",
    "track_id",
    "track_name",
]

app = Flask(__name__, template_folder="html", static_folder="css", static_url_path="/static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "lastfm-demo-secret-key")


def is_valid_userid(userid: str) -> bool:
    if not userid:
        return False
    user_ids = load_user_ids(limit=50000)
    return userid in user_ids


def slugify(text: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text).strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "group"


def load_login_accounts() -> List[dict]:
    behavior = load_behavior_groups()
    if behavior.empty:
        return [{"account_id": "custom::me", "label": "My Custom Account", "source_userid": "", "behavior_group": "Custom", "is_custom": True}]

    all_users = sorted(behavior["userid"].dropna().astype(str).unique().tolist())
    groups = sorted(behavior["behavior_group"].dropna().astype(str).unique().tolist())

    accounts = []
    for group in groups:
        group_users = sorted(behavior[behavior["behavior_group"] == group]["userid"].dropna().astype(str).unique().tolist())
        if not group_users:
            group_users = all_users[:]
        if not group_users:
            continue

        for i in range(5):
            uid = group_users[i % len(group_users)]
            account_id = f"group::{slugify(group)}::{i+1}"
            accounts.append(
                {
                    "account_id": account_id,
                    "label": f"{group} #{i+1} ({uid})",
                    "source_userid": uid,
                    "behavior_group": group,
                    "is_custom": False,
                }
            )

    accounts.append(
        {
            "account_id": "custom::me",
            "label": "My Custom Account",
            "source_userid": "",
            "behavior_group": "Custom",
            "is_custom": True,
        }
    )

    return accounts


def load_account_histories() -> dict:
    if not ACCOUNT_HISTORY_PATH.exists():
        return {}
    try:
        with ACCOUNT_HISTORY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_account_histories(histories: dict) -> None:
    ACCOUNT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ACCOUNT_HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2, ensure_ascii=False)


def get_committed_songs(account_id: str) -> List[dict]:
    histories = load_account_histories()
    rows = histories.get(account_id, [])
    if not isinstance(rows, list):
        return []
    return rows


def commit_song_for_account(account_id: str, artist_name: str, track_name: str) -> bool:
    artist_name = (artist_name or "").strip()
    track_name = (track_name or "").strip()
    if not account_id or not artist_name or not track_name:
        return False

    histories = load_account_histories()
    rows = histories.get(account_id, [])
    if not isinstance(rows, list):
        rows = []

    exists = any(
        str(item.get("artist_name", "")).strip().lower() == artist_name.lower()
        and str(item.get("track_name", "")).strip().lower() == track_name.lower()
        for item in rows
    )
    if exists:
        return False

    rows.insert(
        0,
        {
            "artist_name": artist_name,
            "track_name": track_name,
            "committed_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    histories[account_id] = rows[:1000]
    save_account_histories(histories)
    return True


def clear_account_history(account_id: str) -> None:
    if not account_id:
        return
    histories = load_account_histories()
    if account_id in histories:
        histories[account_id] = []
        save_account_histories(histories)


def collect_top_tracks_from_users(user_ids: List[str], per_user_limit: int = 400, max_tracks: int = 100) -> List[dict]:
    track_counts = {}

    for uid in user_ids:
        hist = get_user_history(uid, limit=per_user_limit)
        if hist.empty:
            continue

        grouped = hist.groupby(["artist_name", "track_name"]).size().reset_index(name="plays")
        for _, row in grouped.iterrows():
            artist_name = str(row["artist_name"]).strip()
            track_name = str(row["track_name"]).strip()
            if not artist_name or not track_name:
                continue
            key = (artist_name, track_name)
            track_counts[key] = track_counts.get(key, 0) + int(row["plays"])

    ranked = sorted(track_counts.items(), key=lambda item: item[1], reverse=True)
    return [{"artist_name": key[0], "track_name": key[1], "plays": score} for key, score in ranked[:max_tracks]]


def build_model_recommendations(model_type: str, source_userid: str, account_id: str, limit: int = 20) -> List[dict]:
    if model_type not in {"clustering", "biclustering"}:
        return []

    committed = get_committed_songs(account_id)
    committed_keys = {
        (
            str(item.get("artist_name", "")).strip().lower(),
            str(item.get("track_name", "")).strip().lower(),
        )
        for item in committed
    }

    peer_user_ids = []
    if source_userid:
        assignments = safe_read_csv(RESULTS_DIR / "user_cluster_assignments.csv")
        if not assignments.empty and {"userid", "kmeans_cluster", "bicluster"}.issubset(assignments.columns):
            assignments = assignments.copy()
            assignments["userid"] = assignments["userid"].astype(str)
            current = assignments[assignments["userid"] == str(source_userid)]
            if not current.empty:
                if model_type == "clustering":
                    label_value = current.iloc[0]["kmeans_cluster"]
                    peers = assignments[assignments["kmeans_cluster"] == label_value]["userid"].astype(str).tolist()
                else:
                    label_value = current.iloc[0]["bicluster"]
                    peers = assignments[assignments["bicluster"] == label_value]["userid"].astype(str).tolist()

                peer_user_ids = [str(source_userid)] + [uid for uid in peers if uid != str(source_userid)]
                peer_user_ids = peer_user_ids[:8]

    candidates = collect_top_tracks_from_users(peer_user_ids, per_user_limit=400, max_tracks=120) if peer_user_ids else []

    recommendations = []
    seen = set()
    for row in candidates:
        artist_name = str(row.get("artist_name", "")).strip()
        track_name = str(row.get("track_name", "")).strip()
        key = (artist_name.lower(), track_name.lower())
        if not artist_name or not track_name or key in committed_keys or key in seen:
            continue
        recommendations.append({"artist_name": artist_name, "track_name": track_name})
        seen.add(key)
        if len(recommendations) >= limit:
            return recommendations

    fallback = build_recommendations(source_userid=source_userid, account_id=account_id, limit=limit * 2)
    for row in fallback:
        artist_name = str(row.get("artist_name", "")).strip()
        track_name = str(row.get("track_name", "")).strip()
        key = (artist_name.lower(), track_name.lower())
        if not artist_name or not track_name or key in committed_keys or key in seen:
            continue
        recommendations.append({"artist_name": artist_name, "track_name": track_name})
        seen.add(key)
        if len(recommendations) >= limit:
            break

    return recommendations


def build_recommendations(source_userid: str, account_id: str, limit: int = 20) -> List[dict]:
    committed = get_committed_songs(account_id)
    committed_keys = {
        (
            str(item.get("artist_name", "")).strip().lower(),
            str(item.get("track_name", "")).strip().lower(),
        )
        for item in committed
    }

    recommendations = []
    seen = set()

    if source_userid:
        user_hist = get_user_history(source_userid, limit=5000)
        if not user_hist.empty:
            top_tracks = (
                user_hist.groupby(["artist_name", "track_name"]).size().reset_index(name="plays").sort_values("plays", ascending=False)
            )
            for _, row in top_tracks.iterrows():
                artist_name = str(row["artist_name"]).strip()
                track_name = str(row["track_name"]).strip()
                key = (artist_name.lower(), track_name.lower())
                if key in committed_keys or key in seen or not artist_name or not track_name:
                    continue
                recommendations.append({"artist_name": artist_name, "track_name": track_name})
                seen.add(key)
                if len(recommendations) >= limit:
                    return recommendations

    if committed and len(recommendations) < limit:
        for item in committed:
            artist = str(item.get("artist_name", "")).strip()
            if not artist:
                continue
            artist_matches = search_song_history(artist, limit=200)
            if artist_matches.empty:
                continue
            artist_tracks = (
                artist_matches.groupby(["artist_name", "track_name"]).size().reset_index(name="plays").sort_values("plays", ascending=False)
            )
            for _, row in artist_tracks.iterrows():
                artist_name = str(row["artist_name"]).strip()
                track_name = str(row["track_name"]).strip()
                key = (artist_name.lower(), track_name.lower())
                if key in committed_keys or key in seen or not artist_name or not track_name:
                    continue
                recommendations.append({"artist_name": artist_name, "track_name": track_name})
                seen.add(key)
                if len(recommendations) >= limit:
                    return recommendations

    if len(recommendations) < limit:
        fallback = search_song_history("", limit=500)
        if not fallback.empty:
            fallback_tracks = (
                fallback.groupby(["artist_name", "track_name"]).size().reset_index(name="plays").sort_values("plays", ascending=False)
            )
            for _, row in fallback_tracks.iterrows():
                artist_name = str(row["artist_name"]).strip()
                track_name = str(row["track_name"]).strip()
                key = (artist_name.lower(), track_name.lower())
                if key in committed_keys or key in seen or not artist_name or not track_name:
                    continue
                recommendations.append({"artist_name": artist_name, "track_name": track_name})
                seen.add(key)
                if len(recommendations) >= limit:
                    break

    return recommendations


@app.before_request
def load_logged_in_user():
    g.current_user = session.get("account_label")


@app.context_processor
def inject_auth_user():
    return {
        "current_user": session.get("account_label"),
        "current_account_id": session.get("account_id"),
        "current_source_userid": session.get("source_userid"),
        "current_behavior_group": session.get("behavior_group"),
    }


def require_login():
    if not session.get("account_id"):
        return redirect(url_for("login", next=request.path))
    return None


def safe_read_json(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_read_csv(path: Path, default=None):
    if default is None:
        default = pd.DataFrame()
    if not path.exists():
        return default
    return pd.read_csv(path)


def load_artifacts():
    analysis = safe_read_json(RESULTS_DIR / "analysis_results.json", {})
    assignments = safe_read_csv(RESULTS_DIR / "user_cluster_assignments.csv")
    migration = safe_read_csv(RESULTS_DIR / "user_cluster_migration.csv")
    crosstab = safe_read_csv(RESULTS_DIR / "user_cluster_crosstab.csv")
    bic_stats = safe_read_csv(RESULTS_DIR / "bicluster_stats.csv")
    cluster_sizes = safe_read_csv(RESULTS_DIR / "cluster_sizes.csv")

    return {
        "analysis": analysis,
        "assignments": assignments,
        "migration": migration,
        "crosstab": crosstab,
        "bicluster_stats": bic_stats,
        "cluster_sizes": cluster_sizes,
    }


def load_behavior_groups() -> pd.DataFrame:
    behavior = safe_read_csv(BEHAVIOR_GROUPS_PATH)
    if behavior.empty:
        return pd.DataFrame(columns=["userid", "behavior_group"])

    required = ["userid", "behavior_group"]
    missing = [col for col in required if col not in behavior.columns]
    if missing:
        return pd.DataFrame(columns=["userid", "behavior_group"])

    behavior = behavior[required].copy()
    behavior["userid"] = behavior["userid"].astype(str)
    return behavior


def add_behavior_group(rows: pd.DataFrame, userid_col: str = "userid") -> pd.DataFrame:
    if rows.empty or userid_col not in rows.columns:
        return rows

    behavior = load_behavior_groups()
    if behavior.empty:
        rows = rows.copy()
        rows["behavior_group"] = "N/A"
        return rows

    merged = rows.copy()
    merged[userid_col] = merged[userid_col].astype(str)
    merged = merged.merge(behavior, how="left", left_on=userid_col, right_on="userid")
    if "userid_y" in merged.columns:
        merged = merged.drop(columns=["userid_y"])
    if "userid_x" in merged.columns:
        merged = merged.rename(columns={"userid_x": userid_col})
    merged["behavior_group"] = merged["behavior_group"].fillna("N/A")
    return merged


def load_user_ids(limit: int = 1000) -> List[str]:
    assignments_path = RESULTS_DIR / "user_cluster_assignments.csv"
    if assignments_path.exists():
        assignments = safe_read_csv(assignments_path)
        if not assignments.empty and "userid" in assignments.columns:
            user_ids = sorted(assignments["userid"].astype(str).dropna().unique().tolist())
            return user_ids[:limit]

    if not LISTENING_PATH.exists():
        return []

    seen = set()
    for chunk in pd.read_csv(
        LISTENING_PATH,
        sep="\t",
        header=None,
        names=LISTENING_COLUMNS,
        usecols=[0],
        chunksize=100000,
        dtype={"userid": "string"},
    ):
        for uid in chunk["userid"].dropna().astype(str).tolist():
            seen.add(uid)
            if len(seen) >= limit:
                return sorted(seen)
    return sorted(seen)


def search_song_history(song_query: str, limit: int = 300) -> pd.DataFrame:
    if not LISTENING_PATH.exists():
        return pd.DataFrame(columns=LISTENING_COLUMNS)

    song_query = (song_query or "").lower()
    matched_chunks = []
    total = 0

    for chunk in pd.read_csv(
        LISTENING_PATH,
        sep="\t",
        header=None,
        names=LISTENING_COLUMNS,
        usecols=[0, 1, 3, 5],
        chunksize=100000,
        dtype="string",
    ):
        if song_query:
            track_match = chunk["track_name"].str.lower().str.contains(song_query, na=False)
            artist_match = chunk["artist_name"].str.lower().str.contains(song_query, na=False)
            matched = chunk[track_match | artist_match]
        else:
            matched = chunk

        if not matched.empty:
            matched_chunks.append(matched)
            total += len(matched)
            if total >= limit:
                break

    if not matched_chunks:
        return pd.DataFrame(columns=["userid", "timestamp", "artist_name", "track_name"])

    result = pd.concat(matched_chunks, ignore_index=True)
    return result.head(limit)


def get_user_history(userid: str, limit: int = 500) -> pd.DataFrame:
    if not LISTENING_PATH.exists() or not userid:
        return pd.DataFrame(columns=["timestamp", "artist_name", "track_name"])

    chunks = []
    total = 0

    for chunk in pd.read_csv(
        LISTENING_PATH,
        sep="\t",
        header=None,
        names=LISTENING_COLUMNS,
        usecols=[0, 1, 3, 5],
        chunksize=100000,
        dtype="string",
    ):
        selected = chunk[chunk["userid"] == userid]
        if not selected.empty:
            chunks.append(selected)
            total += len(selected)
            if total >= limit:
                break

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "artist_name", "track_name"])

    result = pd.concat(chunks, ignore_index=True).head(limit)
    result = result[["timestamp", "artist_name", "track_name"]]
    return result


@app.route("/")
def home():
    if session.get("account_id"):
        account_id = session.get("account_id", "")
        committed_rows = get_committed_songs(account_id)
        return render_template(
            "main.html",
            committed_count=len(committed_rows),
            source_userid=session.get("source_userid", ""),
            behavior_group=session.get("behavior_group", ""),
        )
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    accounts = load_login_accounts()
    account_map = {row["account_id"]: row for row in accounts}

    if request.method == "POST":
        account_id = request.form.get("account_id", "").strip()
        custom_name = request.form.get("custom_name", "").strip()

        if account_id not in account_map:
            flash("Invalid account. Please choose from the list.")
            return render_template("login.html", accounts=accounts, selected_account_id=account_id, custom_name=custom_name)

        account = account_map[account_id]
        if account.get("is_custom"):
            final_name = custom_name if custom_name else "me"
            final_name = slugify(final_name)
            final_account_id = f"custom::{final_name}"
            session["account_id"] = final_account_id
            session["account_label"] = f"Custom Account ({final_name})"
            session["source_userid"] = ""
            session["behavior_group"] = "Custom"
        else:
            session["account_id"] = account["account_id"]
            session["account_label"] = account["label"]
            session["source_userid"] = account["source_userid"]
            session["behavior_group"] = account["behavior_group"]

        session.pop("main_recommendations", None)

        flash(f"Logged in as {session.get('account_label')}")
        next_url = request.args.get("next")
        if next_url:
            return redirect(next_url)
        return redirect(url_for("home"))

    if session.get("account_id"):
        return redirect(url_for("home"))

    return render_template("login.html", accounts=accounts, selected_account_id="", custom_name="")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully")
    return redirect(url_for("login"))


@app.route("/songs")
def song_search():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response

    query = request.args.get("q", "").strip()
    songs = search_song_history(query, limit=300)
    if not songs.empty:
        songs = songs[["artist_name", "track_name"]]
    rows = [] if songs.empty else songs.to_dict(orient="records")

    return render_template(
        "song_search.html",
        search_query=query,
        rows=rows,
        showing_default=not bool(query),
        data_available=LISTENING_PATH.exists(),
    )


@app.route("/commit-song", methods=["POST"])
def commit_song():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response

    account_id = session.get("account_id", "")
    artist_name = request.form.get("artist_name", "").strip()
    track_name = request.form.get("track_name", "").strip()
    redirect_to = request.form.get("redirect_to", "songs").strip()

    committed = commit_song_for_account(account_id, artist_name, track_name)
    if committed:
        flash(f"Added to listening history: {artist_name} - {track_name}")
    else:
        flash("Song already exists in listening history or invalid input")

    if redirect_to == "main":
        return redirect(url_for("home"))
    if redirect_to == "my_history":
        return redirect(url_for("my_history"))
    if redirect_to == "model_clustering":
        return redirect(url_for("model_clustering_page"))
    if redirect_to == "model_biclustering":
        return redirect(url_for("model_biclustering_page"))
    return redirect(url_for("song_search", q=request.form.get("q", "")))


@app.route("/remove-history", methods=["POST"])
def remove_history():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response

    account_id = session.get("account_id", "")
    clear_account_history(account_id)
    flash("Listening history removed for this account")

    redirect_to = request.form.get("redirect_to", "main").strip()
    if redirect_to == "my_history":
        return redirect(url_for("my_history"))
    return redirect(url_for("home"))


@app.route("/history")
def user_history():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response

    selected_user = request.args.get("userid", "").strip()
    user_ids = load_user_ids(limit=1500)

    history_rows = []
    selected_behavior_group = None
    if selected_user:
        history = get_user_history(selected_user, limit=500)
        history_rows = [] if history.empty else history.to_dict(orient="records")
        behavior = load_behavior_groups()
        if not behavior.empty:
            matched = behavior[behavior["userid"] == selected_user]
            if not matched.empty:
                selected_behavior_group = matched.iloc[0]["behavior_group"]

    return render_template(
        "user_history.html",
        user_ids=user_ids,
        selected_user=selected_user,
        selected_behavior_group=selected_behavior_group,
        history_rows=history_rows,
        data_available=LISTENING_PATH.exists(),
    )


@app.route("/my-history")
def my_history():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response

    selected_user = session.get("source_userid", "")
    history = get_user_history(selected_user, limit=500)
    history_rows = [] if history.empty else history.to_dict(orient="records")
    committed_rows = get_committed_songs(session.get("account_id", ""))

    selected_behavior_group = None
    behavior = load_behavior_groups()
    if not behavior.empty:
        matched = behavior[behavior["userid"] == selected_user]
        if not matched.empty:
            selected_behavior_group = matched.iloc[0]["behavior_group"]

    return render_template(
        "user_history.html",
        user_ids=load_user_ids(limit=1500),
        selected_user=selected_user,
        selected_behavior_group=selected_behavior_group,
        history_rows=history_rows,
        committed_rows=committed_rows,
        data_available=LISTENING_PATH.exists(),
    )


@app.route("/model/clustering")
def model_clustering_page():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response

    account_id = session.get("account_id", "")
    source_userid = session.get("source_userid", "")
    rows = build_model_recommendations(
        model_type="clustering",
        source_userid=source_userid,
        account_id=account_id,
        limit=20,
    )
    return render_template(
        "model_page.html",
        model_title="Clustering Model",
        rows=rows,
        redirect_to="model_clustering",
    )


@app.route("/model/biclustering")
def model_biclustering_page():
    redirect_response = require_login()
    if redirect_response:
        return redirect_response

    account_id = session.get("account_id", "")
    source_userid = session.get("source_userid", "")
    rows = build_model_recommendations(
        model_type="biclustering",
        source_userid=source_userid,
        account_id=account_id,
        limit=20,
    )
    return render_template(
        "model_page.html",
        model_title="Bi-Clustering Model",
        rows=rows,
        redirect_to="model_biclustering",
    )


@app.route("/user/<userid>")
def user_detail(userid: str):
    redirect_response = require_login()
    if redirect_response:
        return redirect_response

    data = load_artifacts()
    assignments = data["assignments"]
    assignments = add_behavior_group(assignments, userid_col="userid")

    if assignments.empty or userid not in assignments["userid"].values:
        abort(404, description=f"User '{userid}' not found in output/results/user_cluster_assignments.csv")

    row = assignments[assignments["userid"] == userid].iloc[0].to_dict()

    same_kmeans = assignments[assignments["kmeans_cluster"] == row["kmeans_cluster"]]["userid"].head(20).tolist()
    same_bicluster = assignments[assignments["bicluster"] == row["bicluster"]]["userid"].head(20).tolist()

    return render_template(
        "user_detail.html",
        user=row,
        same_kmeans=same_kmeans,
        same_bicluster=same_bicluster,
    )


@app.route("/plots/<path:filename>")
def plot_file(filename: str):
    file_path = PLOTS_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        abort(404)
    return send_from_directory(PLOTS_DIR, filename)


@app.errorhandler(404)
def not_found(err):
    return render_template("404.html", message=str(err)), 404


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
