"""
Streamlit labeling UI for toxicity posts pulled from GCS.

Run:
  streamlit run scripts/label_ui.py

Each label is POSTed to the running API's `/labels` endpoint (LabeledPost
schema in api/app.py) so it feeds the same GCS label store the API writes to.
If the API is unreachable, the label is still appended to data/labeled.csv
as a local backup and can be resubmitted later.

Note: the API's LabeledPost schema has no field for PII corrections yet, so
the PII call captured here is only kept in the local CSV, not sent to /labels.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
from google.cloud import storage

GCS_BUCKET = "tox-labeled-data"
GCS_PREFIX = "posts"
OUTPUT_PATH = Path("data/labeled.csv")
DEFAULT_API_URL = "http://localhost:8000"


def load_from_gcs(days: int, lang: str | None, sample: int | None) -> pd.DataFrame:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    dates = [(datetime.now(tz=UTC) - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    # Fetch 10x the sample size to allow for stratified sampling headroom; default 4000
    fetch_limit = (sample * 10) if sample else 4000

    rows = []
    done = False
    for date in dates:
        if done:
            break
        prefix = f"{GCS_PREFIX}/dt={date}/"
        for blob in bucket.list_blobs(prefix=prefix):
            for line in blob.download_as_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scores = record.pop("scores", {})
                record.update({f"score_{k}": v for k, v in scores.items()})
                details = record.pop("details", {})
                record.update({f"details_{k}": v for k, v in details.items()})
                rows.append(record)
            if len(rows) >= fetch_limit:
                done = True
                break

    df = pd.DataFrame(rows)

    if lang:
        df = df[df["langs"].apply(lambda x: lang in (x or []))]

    if sample and sample < len(df):
        df["_bucket"] = pd.cut(
            df["score_toxicity"].fillna(0.0).astype(float),
            bins=[0.0, 0.3, 0.7, 1.0],
            labels=["low", "medium", "high"],
            include_lowest=True,
        )
        df = (
            df.groupby("_bucket", observed=True)
            .apply(lambda g: g.sample(min(len(g), max(1, sample // 3)), random_state=42))
            .reset_index(drop=True)
            .drop(columns=["_bucket"])
        )

    return df.reset_index(drop=True)


def load_existing_labels() -> dict:
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        return dict(zip(existing["uri"], existing["toxicity_label"]))
    return {}


def submit_label_to_api(api_url: str, payload: dict) -> tuple[bool, str]:
    try:
        resp = httpx.post(f"{api_url.rstrip('/')}/labels", json=payload, timeout=5.0)
        resp.raise_for_status()
        return True, "ok"
    except Exception as e:
        return False, str(e)


def save_label(row: pd.Series, api_url: str, toxicity: str, hatespeech: str, pii: str) -> bool:
    """Build the LabeledPost payload, POST it to the API, and always keep a local CSV backup."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tox_score = float(row.get("score_toxicity") or 0.0)
    sentiment_score = float(row.get("score_sentiment") or 0.0)
    hatespeech_score = float(row.get("score_hatespeech") or 0.0)
    labeled_at = datetime.now(UTC).isoformat()

    api_payload = {
        "uri": str(row.get("uri", "")),
        "text": str(row.get("text", "")),
        "author_handle": str(row.get("author_handle", "")),
        "created_at": str(row.get("created_at", "")),
        "feed_uri": str(row.get("feed_uri", "")),
        "feed_name": str(row.get("feed_name", "")),
        "toxicity_score": tox_score,
        "toxicity_label": int(row.get("label", tox_score >= 0.5)),
        "sentiment_score": sentiment_score,
        "hatespeech_score": hatespeech_score,
        "corrected_toxicity_label": 1 if toxicity == "toxic" else 0,
        "corrected_hatespeech_label": 1 if hatespeech == "hate_speech" else 0,
        "tags": [],
        "labeled_at": labeled_at,
    }

    submitted, detail = submit_label_to_api(api_url, api_payload)

    entry = {
        **api_payload,
        "pii_label": pii,
        "score_pii": row.get("score_pii", None),
        "details_pii": ",".join(row.get("details_pii") or []) if row.get("details_pii") else "",
        "model_version": row.get("model_version", None),
        "api_submitted": submitted,
        "api_error": "" if submitted else detail,
    }
    df_new = pd.DataFrame([entry])
    if OUTPUT_PATH.exists():
        df_new.to_csv(OUTPUT_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(OUTPUT_PATH, index=False)

    return submitted


# ── Sidebar controls ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Toxicity Labeler", layout="centered")
st.title("Toxicity Labeler")

with st.sidebar:
    st.header("Data settings")
    days = st.number_input("Days back", min_value=1, max_value=365, value=30)
    lang = st.text_input("Language filter", value="en")
    sample = st.number_input("Sample size", min_value=10, max_value=5000, value=400)
    load_btn = st.button("Load data")

    st.header("API settings")
    api_url = st.text_input("Labeling API URL", value=DEFAULT_API_URL)
    st.caption("Labels POST to `{api_url}/labels`. If unreachable, they're still saved to data/labeled.csv.")

# ── Session state ─────────────────────────────────────────────────────────────

if "df" not in st.session_state:
    st.session_state.df = None
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "labels" not in st.session_state:
    st.session_state.labels = load_existing_labels()

if load_btn:
    status = st.status("Loading from GCS...", expanded=True)
    with status:
        st.write("Connecting to GCS...")
        df = load_from_gcs(days, lang or None, sample)
        st.write(f"Fetched {len(df)} records, applying filters...")
        already = set(st.session_state.labels.keys())
        df = df[~df["uri"].isin(already)].reset_index(drop=True)
        status.update(label=f"Loaded {len(df)} unlabeled records", state="complete")
    st.session_state.df = df
    st.session_state.idx = 0

# ── Labeling UI ───────────────────────────────────────────────────────────────

df = st.session_state.df

if df is None:
    st.info("Configure settings in the sidebar and click **Load data** to begin.")
    st.stop()

if len(df) == 0:
    st.success("All records labeled!")
    st.stop()

idx = st.session_state.idx
if idx >= len(df):
    st.success(f"Done! Labeled {len(st.session_state.labels)} records total. Saved to `{OUTPUT_PATH}`.")
    st.stop()

row = df.iloc[idx]
progress = idx / len(df)

st.progress(progress, text=f"{idx} / {len(df)}")

st.subheader("Post")
st.write(row.get("text", ""))

col1, col2, col3 = st.columns(3)
with col1:
    tox_score = row.get("score_toxicity", None)
    st.metric("Model toxicity", f"{tox_score:.3f}" if tox_score is not None else "—")
with col2:
    sent_score = row.get("score_sentiment", None)
    st.metric("Sentiment", f"{sent_score:.3f}" if sent_score is not None else "—")
with col3:
    hs_score = row.get("score_hatespeech", None)
    st.metric("Model hate speech", f"{hs_score:.3f}" if hs_score is not None else "—")

pii_score = row.get("score_pii", None)
pii_entities = row.get("details_pii", None)
if pii_score is not None:
    pii_note = "PII detected" if pii_score >= 1.0 else "No PII detected"
    if pii_entities:
        pii_note += f" ({', '.join(pii_entities)})"
    st.caption(f"Model PII signal: {pii_note}")

st.divider()

toxicity = st.radio("Toxicity", ["toxic", "not_toxic"], horizontal=True)
hatespeech = st.radio("Hate speech", ["hate_speech", "non_hate_speech"], horizontal=True)
pii = st.radio("PII", ["contains_pii", "no_pii"], horizontal=True)

col_save, col_skip = st.columns(2)


def apply_label() -> None:
    submitted = save_label(row, api_url, toxicity, hatespeech, pii)
    st.session_state.labels[row.get("uri", "")] = toxicity
    st.session_state.idx += 1
    if not submitted:
        st.session_state["_last_api_warning"] = True


with col_save:
    if st.button("Save label", type="primary", use_container_width=True):
        apply_label()
        st.rerun()

with col_skip:
    if st.button("Skip", use_container_width=True):
        st.session_state.idx += 1
        st.rerun()

if st.session_state.pop("_last_api_warning", False):
    st.warning("Could not reach the labeling API — label was saved to data/labeled.csv only.")
