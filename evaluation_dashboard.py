import streamlit as st
import pandas as pd
import numpy as np

from evaluation import Evaluator
from content_based import ContentBasedGymRecommender
from knowledge_based import GymRecommender as KBRecommender
from hybrid_recommender import HybridGymRecommender


# ============================================================
# CONFIG
# ============================================================
GYMS_PATH = "final-final/gyms_cleaned.csv"
USERS_PATH = "final-final/users_cleaned.csv"
CHECKINS_PATH = "final-final/checkin_checkout_history_expanded.csv"
GYNAMES_PATH = "final-final/gym_names.csv"

TOP_K = 5
FAST_SAMPLE = 50


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Gym RS – Evaluation Dashboard", layout="wide")
st.title("🏋️‍♂️ Recommendation System Evaluation Dashboard")
st.write("Evaluate **Content-Based**, **Knowledge-Based**, and **Hybrid** recommenders.")


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    gyms = pd.read_csv(GYMS_PATH)
    users = pd.read_csv(USERS_PATH)
    checkins = pd.read_csv(CHECKINS_PATH)
    return gyms, users, checkins

gyms, users, checkins = load_data()


# ============================================================
# GROUND TRUTH BUILDER
# ============================================================
def build_ground_truth():
    gt = {}
    for user_id, df_u in checkins.groupby("user_id"):
        top = df_u["gym_id"].value_counts().head(5).index.tolist()
        gt[str(user_id)] = [str(x).strip() for x in top]
    return gt

ground_truth = build_ground_truth()


# ============================================================
# FAST/EVAL MODE
# ============================================================
st.sidebar.header("⚡ Speed Settings")

fast_mode = st.sidebar.checkbox("Enable FAST mode", value=True)
sample_size = st.sidebar.slider("Number of users", 10, 200, 30)

if fast_mode:
    eval_users = users["user_id"].sample(sample_size, random_state=42).tolist()
    st.sidebar.success(f"Evaluating {sample_size} users (FAST mode).")
else:
    eval_users = users["user_id"].tolist()
    st.sidebar.warning("Evaluating ALL users (slow).")


# ============================================================
# INITIALIZE MODELS
# ============================================================
@st.cache_resource
def build_recommenders():
    kb = KBRecommender(GYMS_PATH, GYNAMES_PATH)
    cb = ContentBasedGymRecommender(gyms_df=gyms, users_df=users, checkins_df=checkins)
    hybrid = HybridGymRecommender(
        kb_recommender=kb,
        cb_recommender=cb,
        hybrid_strategy="cascade",
        kb_weight=0.4,
        cb_weight=0.6
    )
    return kb, cb, hybrid

kb_model, cb_model, hybrid_model = build_recommenders()


# ============================================================
# RECOMMENDER WRAPPERS (IMPORTANT!)
# ============================================================
def cb_recommend(uid):
    """Content-Based returns a LIST of dicts."""
    try:
        rows = cb_model.recommend(user_id=uid, top_k=TOP_K, diversity_factor=0.3)
        return [str(r["gym_id"]).strip() for r in rows]
    except:
        return []


def kb_recommend(uid):
    """Knowledge-Based uses recommend_with_utility_function() → returns DF."""
    try:
        u = users[users["user_id"] == uid].iloc[0]
        df = kb_model.recommend_with_utility_function(
            city=u.get("city"),
            gender=u.get("gender", "Male"),
            min_price=0,
            max_price=120,
            age=int(u.get("age", 25)),
            desired_facilities=[],
            preferred_city=u.get("city"),
            weights=None
        )
        return [str(g).strip() for g in df["gym_id"].head(TOP_K).tolist()]
    except:
        return []


def hybrid_recommend(uid):
    """Hybrid returns a DataFrame."""
    try:
        u = users[users["user_id"] == uid].iloc[0]
        df = hybrid_model.recommend(
            user_id=uid,
            city=u.get("city"),
            gender=u.get("gender", "Male"),
            min_price=0,
            max_price=120,
            age=int(u.get("age", 25)),
            desired_facilities=[],
            preferred_city=u.get("city"),
            top_k=TOP_K
        )
        return [str(g).strip() for g in df["gym_id"].tolist()]
    except:
        return []


# ============================================================
# RUN EVALUATION
# ============================================================
if st.sidebar.button("Run Evaluation"):
    st.info("Running evaluation… please wait ⏳")

    cb_recs = {str(uid): cb_recommend(uid) for uid in eval_users}
    kb_recs = {str(uid): kb_recommend(uid) for uid in eval_users}
    hybrid_recs = {str(uid): hybrid_recommend(uid) for uid in eval_users}

    evaluator = Evaluator()

    cb_results = evaluator.evaluate(cb_recs, ground_truth, TOP_K)
    kb_results = evaluator.evaluate(kb_recs, ground_truth, TOP_K)
    hybrid_results = evaluator.evaluate(hybrid_recs, ground_truth, TOP_K)

    # ============================================================
    # GLOBAL METRICS
    # ============================================================
    st.header("📊 Global Comparison Table")

    def avg(df):
        return df.select_dtypes("number").mean()

    comparison = pd.DataFrame({
        "Content-Based": avg(cb_results),
        "Knowledge-Based": avg(kb_results),
        "Hybrid": avg(hybrid_results)
    })

    st.dataframe(comparison.style.format("{:.3f}"))

    st.header("📈 Metric Comparison Chart")
    st.bar_chart(comparison)

    # ============================================================
    # USER-LEVEL RESULTS
    # ============================================================
    st.header("🔍 User-Level Metrics")

    user = st.selectbox("Select User", options=[str(u) for u in eval_users])

    st.write("### Content-Based")
    st.dataframe(cb_results[cb_results["user_id"] == user])

    st.write("### Knowledge-Based")
    st.dataframe(kb_results[kb_results["user_id"] == user])

    st.write("### Hybrid")
    st.dataframe(hybrid_results[hybrid_results["user_id"] == user])

    # ============================================================
    # EXPORT
    # ============================================================
    st.header("📥 Download Results")

    st.download_button(
        "Download aggregated metrics",
        comparison.to_csv().encode("utf-8"),
        file_name="evaluation_results.csv",
        mime="text/csv"
    )

else:
    st.info("Click **Run Evaluation** to start.")
