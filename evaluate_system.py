import pandas as pd
from evaluation import Evaluator

from content_based import ContentBasedGymRecommender
from knowledge_based import GymRecommender
from hybrid_recommender import HybridGymRecommender


# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
gyms = pd.read_csv("final-final/gyms_cleaned.csv")
users = pd.read_csv("final-final/users_cleaned.csv")
checkins = pd.read_csv("final-final/checkin_checkout_history_expanded.csv")


# ------------------------------------------------------
# Build Ground Truth (Top-5 Gyms per User)
# ------------------------------------------------------
ground_truth = {}

for user_id, df_user in checkins.groupby("user_id"):
    top_gyms = (
        df_user["gym_id"]
        .value_counts()
        .head(5)
        .index
        .tolist()
    )
    ground_truth[str(user_id)] = [str(g) for g in top_gyms]


# ------------------------------------------------------
# Initialize Recommenders
# ------------------------------------------------------
kb = GymRecommender(
   "final-final/gyms_cleaned.csv",
   "final-final/gym_names.csv"
)

cb = ContentBasedGymRecommender(
    gyms_df=gyms,
    users_df=users,
    checkins_df=checkins
)

hybrid = HybridGymRecommender(
    kb_recommender=kb,
    cb_recommender=cb,
    hybrid_strategy="cascade",  # or "weighted"
    kb_weight=0.4,
    cb_weight=0.6
)


# ------------------------------------------------------
# Helper: Generate Recommendations for an RS
# ------------------------------------------------------
def get_recommendations_for_all(rs, label):
    recommendations = {}
    print(f"\nGenerating recommendations using: {label}")

    for user_id in users["user_id"].unique():
        # Important: ensure everything converted to str
        try:
            rec_df = rs.recommend(
                user_id=user_id,
                city="San Francisco",
                gender="Male",
                age=25,
                min_price=0,
                max_price=120,
                desired_facilities=["Sauna"],
                top_k=5
            )

            gym_ids = [str(x) for x in rec_df["gym_id"].tolist()]
            recommendations[str(user_id)] = gym_ids

        except Exception as e:
            print(f"⚠️ Recommender {label} failed for user {user_id}: {e}")
            recommendations[str(user_id)] = []

    return recommendations


# ------------------------------------------------------
# Generate Recommendations for CB, KB, Hybrid
# ------------------------------------------------------
cb_recs = get_recommendations_for_all(cb, "CONTENT-BASED")
kb_recs = get_recommendations_for_all(kb, "KNOWLEDGE-BASED")
hybrid_recs = get_recommendations_for_all(hybrid, "HYBRID")


# ------------------------------------------------------
# Evaluate Each Recommender
# ------------------------------------------------------
evaluator = Evaluator()

cb_results = evaluator.evaluate(cb_recs, ground_truth, k=5)
kb_results = evaluator.evaluate(kb_recs, ground_truth, k=5)
hybrid_results = evaluator.evaluate(hybrid_recs, ground_truth, k=5)


# ------------------------------------------------------
# Display Final Comparative Results
# ------------------------------------------------------
print("\n====================== FINAL EVALUATION ======================\n")

comparison = pd.DataFrame({
    "Content-Based": cb_results.mean(),
    "Knowledge-Based": kb_results.mean(),
    "Hybrid": hybrid_results.mean(),
})

print(comparison)

print("\n===============================================================\n")
