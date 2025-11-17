import pandas as pd
from knowledge_based import GymRecommender as KBRecommender
from content_based import ContentBasedGymRecommender
from hybrid_recommender import HybridGymRecommender   # your hybrid file


def load_data():
    print("🔄 Loading data...")

    gyms = pd.read_csv("final-final/gyms_cleaned.csv")
    users = pd.read_csv("final-final/users_cleaned.csv")
    checkins = pd.read_csv("final-final/checkin_checkout_history_expanded.csv")
    gym_names = pd.read_csv("final-final/gym_names.csv")

    # Ensure gym_name exists
    if "gym_name" not in gyms.columns:
        gyms = gyms.merge(gym_names[['gym_id', 'gym_name']], on="gym_id", how="left")
        gyms['gym_name'] = gyms['gym_name'].fillna("Unnamed Gym")

    print("✅ Data loaded successfully")
    print(f"Gyms: {gyms.shape}, Users: {users.shape}, Checkins: {checkins.shape}")

    return gyms, users, checkins


def main():

    gyms, users, checkins = load_data()

    # Pick a random user for testing
    user_id = users['user_id'].iloc[0]
    user = users[users['user_id'] == user_id].iloc[0]

    print(f"\n👤 Testing hybrid recommendations for user: {user_id}")
    print(f"User city = {user['city']} | age = {user['age']}\n")

    # ---------------------------
    # 1) KB RECOMMENDER
    # ---------------------------
    kb = KBRecommender(
        "final-final/gyms_cleaned.csv",
        "final-final/gym_names.csv"
    )

    # ---------------------------
    # 2) CONTENT-BASED RECOMMENDER
    # ---------------------------
    cb = ContentBasedGymRecommender(
        gyms_df=gyms,
        users_df=users,
        checkins_df=checkins
    )

    # ---------------------------
    # 3) HYBRID RECOMMENDER
    # ---------------------------
    hybrid = HybridGymRecommender(
        kb_recommender=kb,
        cb_recommender=cb,
        hybrid_strategy="cascade",   # cascade | weighted | switching
        kb_weight=0.4,
        cb_weight=0.6,
    )

    # -----------------------------------------
    # RUN HYBRID RECOMMENDATION
    # -----------------------------------------
    results = hybrid.recommend(
        user_id=user_id,
        city=user["city"],
        gender=user["gender"] if "gender" in user else "Male",
        min_price=0,
        max_price=80,
        age=int(user["age"]),
        desired_facilities=["Yoga Studio", "Swimming Pool"],
        preferred_city=user["city"],
        top_k=5
    )
    final_results = llm_reranker.rerank(
    candidates=hybrid_candidates,
    user_row=user,
    desired_facilities=["Yoga Studio", "Swimming Pool"],
    budget_min=0,
    budget_max=80
)

    print("\n" + "=" * 60)
    print("🏆 HYBRID RECOMMENDATIONS")
    print("=" * 60)

    if results.empty:
        print("⚠ No recommendations found!")
        return

    for i, (_, gym) in enumerate(final_results.iterrows(), 1):
        print(f"\n{i}. {gym['gym_name']}")
        print(f"   → Type: {gym.get('gym_type', 'N/A')}")
        print(f"   → Price: {gym.get('final_price', 'N/A')}")
        print(f"   → KB Utility: {gym.get('utility_score', 0):.3f}")
        print(f"   → CB Similarity: {gym.get('cb_similarity_score', 0):.3f}")
        print(f"   → Hybrid Score: {gym['hybrid_score']:.3f}")
        print(f"   → Explanation: {gym['hybrid_explanation']}")


if __name__ == "__main__":
    main()
