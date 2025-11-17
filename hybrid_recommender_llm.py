# Hybrid Recommendation System


import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from llm_as_a_reranker import LLMReranker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HybridGymRecommenderLLM:
    """
    Hybrid recommender that combines:
    1. Knowledge-Based filtering (hard constraints)
    2. Content-Based ranking (personalization)
    """
    
    def __init__(self, kb_recommender, cb_recommender, 
                 hybrid_strategy: str = 'cascade',
                 kb_weight: float = 0.4,
                 cb_weight: float = 0.6):
        """
        Initialize hybrid recommender.
        
        Args:
            kb_recommender: Knowledge-based recommender instance
            cb_recommender: Content-based recommender instance
            hybrid_strategy: 'cascade', 'weighted', or 'switching'
            kb_weight: Weight for KB scores (if weighted strategy)
            cb_weight: Weight for CB scores (if weighted strategy)
        """
        self.kb_recommender = kb_recommender
        self.cb_recommender = cb_recommender
        self.hybrid_strategy = hybrid_strategy
        self.llm = LLMReranker(model="llama3.2:1b")
        self.kb_weight = kb_weight
        self.cb_weight = cb_weight
        
        # Normalize weights
        total = kb_weight + cb_weight
        self.kb_weight = kb_weight / total
        self.cb_weight = cb_weight / total
        
        logger.info(f"Hybrid Strategy: {hybrid_strategy}")
        logger.info(f"Weights - KB: {self.kb_weight:.2f}, CB: {self.cb_weight:.2f}")
    
    def recommend(self, 
                  user_id: str,
                  city: str,
                  gender: str,
                  min_price: float,
                  max_price: float,
                  age: int,
                  desired_facilities: List[str],
                  preferred_city: str = None,
                  weights: Dict[str, float] = None,
                  top_k: int = 5) -> pd.DataFrame:
        """
        Generate hybrid recommendations.
        
        Strategy:
        1. CASCADE: KB filters → CB ranks
        2. WEIGHTED: Combine KB utility + CB similarity scores
        3. SWITCHING: Choose method based on user profile
        """
        
        if self.hybrid_strategy == 'cascade':
            return self._cascade_recommend(
                user_id, city, gender, min_price, max_price, age,
                desired_facilities, preferred_city, weights, top_k
            )
        elif self.hybrid_strategy == 'weighted':
            return self._weighted_recommend(
                user_id, city, gender, min_price, max_price, age,
                desired_facilities, preferred_city, weights, top_k
            )
        elif self.hybrid_strategy == 'switching':
            return self._switching_recommend(
                user_id, city, gender, min_price, max_price, age,
                desired_facilities, preferred_city, weights, top_k
            )
        else:
            raise ValueError(f"Unknown strategy: {self.hybrid_strategy}")
    
    def _cascade_recommend(self, user_id, city, gender, min_price, max_price,
                          age, desired_facilities, preferred_city, weights, top_k):
        """
        CASCADE STRATEGY:
        1. Use KB to filter gyms by hard constraints
        2. Use CB to rank filtered gyms by personalization
        """
        logger.info("\n" + "="*70)
        logger.info("HYBRID RECOMMENDATION - CASCADE STRATEGY")
        logger.info("="*70)
        
        # STEP 1: Knowledge-Based Filtering
        logger.info("\n[STEP 1] Applying KB filters (constraints)...")
        kb_results = self.kb_recommender.recommend_with_utility_function(
            city=city,
            gender=gender,
            min_price=min_price,
            max_price=max_price,
            age=age,
            desired_facilities=desired_facilities,
            preferred_city=preferred_city,
            weights=weights
        )
        
        if kb_results.empty:
            logger.warning("No gyms passed KB filters!")
            return pd.DataFrame()
        
        logger.info(f"KB filtered: {len(kb_results)} candidate gyms")
        
        # STEP 2: Content-Based Ranking
        logger.info("\n[STEP 2] Applying CB ranking (personalization)...")
        
        # Get CB recommendations for this user (filtered by same city)
        cb_recommendations = self.cb_recommender.recommend(
            user_id=user_id,
            top_k=len(kb_results),  # Rank all KB candidates
            diversity_factor=0.2
        )
        
        # Create CB score lookup
        cb_scores = {
            rec['gym_id']: {
                'cb_score': rec['similarity_score'],
                'cb_explanation': rec['explanation']
            }
            for rec in cb_recommendations
        }
        # STEP 3: LLM Semantic Re-Ranking
        logger.info("\n[STEP 3] Applying LLM semantic reranking...")

        def llm_score_row(row):
            user_profile = {
                "age": age,
                "gender": gender,
                "city": city,
                "desired_facilities": desired_facilities,
                "history": "Summary of past workouts from check-ins"
            }
            gym_profile = {
                "gym_name": row.get("gym_name", row["gym_id"]),
                "gym_type": row["gym_type"],
                "city": row["city"],
                "price": row["final_price"],
                "facilities": row["facilities"],
                "description": row.get("description", "")
            }
            return self.llm.score(user_profile, gym_profile)

        kb_results["llm_score"] = kb_results.apply(llm_score_row, axis=1)

        # Merge KB utility with CB similarity
        kb_results['cb_similarity_score'] = kb_results['gym_id'].map(
            lambda gid: cb_scores.get(gid, {}).get('cb_score', 0.0)
        )
        kb_results['cb_explanation'] = kb_results['gym_id'].map(
            lambda gid: cb_scores.get(gid, {}).get('cb_explanation', '')
        )
        
        # Hybrid score: Weighted combination
        kb_results["hybrid_score"] = (
        0.4 * kb_results["utility_score"]
        + 0.4 * kb_results["cb_similarity_score"]
        + 0.2 * kb_results["llm_score"]
        )

        
        # Re-rank by hybrid score
        kb_results = kb_results.sort_values('hybrid_score', ascending=False)
        
        # Generate hybrid explanations
        kb_results['hybrid_explanation'] = kb_results.apply(
            lambda row: self._generate_hybrid_explanation(
                row, desired_facilities, age
            ),
            axis=1
        )
        
        logger.info(f"Final recommendations: {min(top_k, len(kb_results))} gyms")
        
        return kb_results.head(top_k)
    
    def _weighted_recommend(self, user_id, city, gender, min_price, max_price,
                           age, desired_facilities, preferred_city, weights, top_k):
        """
        WEIGHTED STRATEGY:
        Compute both KB and CB scores for ALL gyms, then combine
        """
        logger.info("\n" + "="*70)
        logger.info("HYBRID RECOMMENDATION - WEIGHTED STRATEGY")
        logger.info("="*70)
        
        # Get KB results (with utility scores)
        kb_results = self.kb_recommender.recommend_with_utility_function(
            city=city,
            gender=gender,
            min_price=min_price,
            max_price=max_price,
            age=age,
            desired_facilities=desired_facilities,
            preferred_city=preferred_city,
            weights=weights
        )
        
        if kb_results.empty:
            return pd.DataFrame()
        
        # Get CB results (with similarity scores)
        cb_recommendations = self.cb_recommender.recommend(
            user_id=user_id,
            top_k=len(kb_results),
            diversity_factor=0.2
        )
        
        cb_scores = {
            rec['gym_id']: rec['similarity_score']
            for rec in cb_recommendations
        }
        
        kb_results['cb_similarity_score'] = kb_results['gym_id'].map(
            lambda gid: cb_scores.get(gid, 0.0)
        )
        
        # Weighted hybrid score
        kb_results['hybrid_score'] = (
            self.kb_weight * kb_results['utility_score'] +
            self.cb_weight * kb_results['cb_similarity_score']
        )
        
        kb_results = kb_results.sort_values('hybrid_score', ascending=False)
        
        kb_results['hybrid_explanation'] = kb_results.apply(
            lambda row: self._generate_hybrid_explanation(
                row, desired_facilities, age
            ),
            axis=1
        )
        
        return kb_results.head(top_k)
    
    def _switching_recommend(self, user_id, city, gender, min_price, max_price,
                            age, desired_facilities, preferred_city, weights, top_k):
        """
        SWITCHING STRATEGY:
        Choose KB or CB based on user profile
        - Cold start users (< 3 checkins): Use KB only
        - Warming users (3-9 checkins): Use KB with light CB influence
        - Warm/Hot users (10+ checkins): Use CB primarily
        """
        logger.info("\n" + "="*70)
        logger.info("HYBRID RECOMMENDATION - SWITCHING STRATEGY")
        logger.info("="*70)
        
        # Get user insights to determine tier
        user_insights = self.cb_recommender.get_user_insights(user_id)
        checkin_count = user_insights['checkin_count']
        tier = user_insights['tier']
        
        logger.info(f"User tier: {tier} ({checkin_count} checkins)")
        
        if tier == 'cold_start':
            logger.info("→ Using KB-dominant approach (cold start)")
            # 80% KB, 20% CB
            self.kb_weight = 0.8
            self.cb_weight = 0.2
        elif tier == 'warming':
            logger.info("→ Using balanced approach (warming)")
            # 50% KB, 50% CB
            self.kb_weight = 0.5
            self.cb_weight = 0.5
        else:
            logger.info("→ Using CB-dominant approach (warm/hot)")
            # 30% KB, 70% CB
            self.kb_weight = 0.3
            self.cb_weight = 0.7
        
        # Use cascade strategy with adjusted weights
        return self._cascade_recommend(
            user_id, city, gender, min_price, max_price, age,
            desired_facilities, preferred_city, weights, top_k
        )
    
    def _generate_hybrid_explanation(self, gym_row, desired_facilities, age):
        """Generate explanation combining KB and CB insights."""
        reasons = []
        
        # Price match - use actual base_price
        base_price = gym_row.get('base_price', gym_row.get('price_per_month', 50.0))
        final_price = gym_row.get('final_price', base_price)
        
        if age < 18:
            reasons.append(f"fits your budget at ${final_price:.0f}/mo (with student discount)")
        else:
            reasons.append(f"fits your budget at ${final_price:.0f}/mo")
        
        # Facility match
        if desired_facilities:
            gym_facilities = [f.strip() for f in str(gym_row.get('facilities', '')).split(',')]
            matched = [f for f in desired_facilities if f in gym_facilities]
            if matched:
                if len(matched) == len(desired_facilities):
                    reasons.append(f"has all {len(matched)} facilities you requested")
                else:
                    reasons.append(f"has {len(matched)}/{len(desired_facilities)} facilities you want")
        
        # CB personalization
        if 'cb_similarity_score' in gym_row and gym_row['cb_similarity_score'] > 0.5:
            reasons.append(f"matches your workout patterns (similarity: {gym_row['cb_similarity_score']:.2f})")
        
        # Hybrid score
        if 'hybrid_score' in gym_row:
            reasons.append(f"overall match score: {gym_row['hybrid_score']:.2f}")
        
        # Limit to top 3 reasons
        reasons = reasons[:3]
        
        gym_name = gym_row.get('gym_name', gym_row['gym_id'])
        explanation = f"{gym_name} is recommended because it {', and '.join(reasons)}."
        
        return explanation


# ======================================================================
# EXAMPLE USAGE
# ======================================================================

if __name__ == "__main__":
    from knowledge_based import GymRecommender as KBRecommender
    
    
    gyms = pd.read_csv("final-final/gyms_cleaned.csv")
    users = pd.read_csv("final-final/users_cleaned.csv")
    checkins = pd.read_csv("final-final/checkin_checkout_history_expanded.csv")
    gym_names = pd.read_csv("final-final/gym_names.csv")
    
    gyms = gyms.merge(gym_names[['gym_id', 'gym_name']], on='gym_id', how='left')
    gyms['gym_name'] = gyms['gym_name'].fillna("Unnamed Gym")
    
    # Initialize both recommenders
    kb_recommender = KBRecommender(
        'final-final/gyms_cleaned.csv',
        'final-final/gym_names.csv'
    )
    
    from content_based import ContentBasedGymRecommender
    cb_recommender = ContentBasedGymRecommender(
        gyms_df=gyms,
        users_df=users,
        checkins_df=checkins
    )
    
    # Initialize hybrid recommender
    hybrid = HybridGymRecommenderLLM(
        kb_recommender=kb_recommender,
        cb_recommender=cb_recommender,
        hybrid_strategy='cascade',  # or 'weighted' or 'switching'
        kb_weight=0.4,
        cb_weight=0.6
    )
    
    # Get recommendations
    user_id = users['user_id'].iloc[0]
    user = users[users['user_id'] == user_id].iloc[0]
    
    recommendations = hybrid.recommend(
        user_id=user_id,
        city=user['city'],
        gender='Male',
        min_price=0,
        max_price=50,
        age=25,
        desired_facilities=['Yoga Studio', 'Swimming Pool', 'Sauna'],
        preferred_city=user['city'],
        top_k=5
    )

    # Display
    print("\n" + "="*80)
    print("HYBRID RECOMMENDATIONS")
    print("="*80)
    
    for idx, (_, gym) in enumerate(recommendations.iterrows(), 1):
        print(f"\n{idx}. {gym.get('gym_name', gym['gym_id'])}")
        print(f"   Type: {gym['gym_type']}")
        print(f"   Price: ${gym['final_price']:.2f}/month")
        print(f"   KB Utility: {gym['utility_score']:.3f}")
        print(f"   CB Similarity: {gym.get('cb_similarity_score', 0):.3f}")
        print(f"   Hybrid Score: {gym['hybrid_score']:.3f}")
        print(f"   Explanation: {gym['hybrid_explanation']}")