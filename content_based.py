import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from collections import Counter
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeatureExtractor:
    """Extracts and transforms features with actual gym prices."""

    def __init__(self):
        self.tfidf_facilities = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b',
            max_features=50,
            min_df=1,
            sublinear_tf=True
        )
        
        self.price_scaler = MinMaxScaler()
        self.gym_type_encoder = {}
        
        # No fixed price map - we'll use actual prices from dataset

    def fit_facilities(self, facilities_series: pd.Series) -> np.ndarray:
        facilities_text = facilities_series.fillna('').astype(str)
        facilities_matrix = self.tfidf_facilities.fit_transform(facilities_text)
        return facilities_matrix.toarray()

    def transform_facilities(self, facilities_series: pd.Series) -> np.ndarray:
        facilities_text = facilities_series.fillna('').astype(str)
        return self.tfidf_facilities.transform(facilities_text).toarray()

    def fit_gym_types(self, gym_types: pd.Series) -> Dict[str, int]:
        unique_types = gym_types.fillna('Standard').str.strip().unique()
        self.gym_type_encoder = {gtype: idx for idx, gtype in enumerate(sorted(unique_types))}
        return self.gym_type_encoder

    def encode_gym_type(self, gym_type: str) -> np.ndarray:
        vector = np.zeros(len(self.gym_type_encoder))
        gym_type = str(gym_type).strip() if pd.notna(gym_type) else 'Standard'
        if gym_type in self.gym_type_encoder:
            vector[self.gym_type_encoder[gym_type]] = 1.0
        else:
            vector[0] = 1.0
        return vector

    def fit_prices(self, prices: pd.Series) -> np.ndarray:
        """Normalize actual base prices to [0, 1] range."""
        prices_clean = prices.fillna(prices.median()).values.reshape(-1, 1)
        self.price_scaler.fit(prices_clean)
        normalized = self.price_scaler.transform(prices_clean)
        logger.info(f"Normalized prices: range ${prices.min():.2f} - ${prices.max():.2f}")
        return normalized.flatten()

    def transform_price(self, price: float) -> float:
        """Transform a single price value."""
        if pd.isna(price):
            price = 50.0
        price_clean = np.array([[price]])
        return self.price_scaler.transform(price_clean)[0, 0]
    
    def get_actual_price(self, price: float) -> float:
        """Return the actual dollar price."""
        return price if pd.notna(price) else 50.0


class GymFeatureBuilder:
    """Builds feature vectors for gyms."""

    def __init__(self, gyms_df: pd.DataFrame, feature_extractor: FeatureExtractor):
        self.gyms_df = gyms_df.copy()  # Make a copy to avoid issues
        self.feature_extractor = feature_extractor
        self.gym_features = {}
        self.feature_dimensions = {}
        
        # Store gym metadata for later use
        self.gym_metadata = self.gyms_df.set_index('gym_id').to_dict('index')

    def build_all_features(self) -> Dict[str, np.ndarray]:
        logger.info("Building gym feature vectors...")
        
        feature_blocks = []
        current_dim = 0

        # 1. Facilities
        facilities_features = self.feature_extractor.fit_facilities(self.gyms_df['facilities'])
        feature_blocks.append(facilities_features)
        facilities_dims = facilities_features.shape[1]
        self.feature_dimensions['facilities'] = (current_dim, current_dim + facilities_dims)
        current_dim += facilities_dims

        # 2. Gym Type
        self.feature_extractor.fit_gym_types(self.gyms_df['gym_type'])
        gym_type_features = np.array([
            self.feature_extractor.encode_gym_type(gt)
            for gt in self.gyms_df['gym_type']
        ])
        feature_blocks.append(gym_type_features)
        type_dims = gym_type_features.shape[1]
        self.feature_dimensions['gym_type'] = (current_dim, current_dim + type_dims)
        current_dim += type_dims

        # 3. Price (from gym type)
        price_features = self.feature_extractor.fit_prices(self.gyms_df['base_price']).reshape(-1, 1)
        feature_blocks.append(price_features)
        self.feature_dimensions['price'] = (current_dim, current_dim + 1)
        current_dim += 1

        full_features = np.hstack(feature_blocks)

        for idx, gym_id in enumerate(self.gyms_df['gym_id']):
            self.gym_features[gym_id] = full_features[idx]

        logger.info(f"Built {full_features.shape[0]} gyms × {full_features.shape[1]} features")
        return self.gym_features

    def get_feature_vector(self, gym_id: str) -> Optional[np.ndarray]:
        return self.gym_features.get(gym_id)
    
    def get_gym_metadata(self, gym_id: str) -> Optional[Dict]:
        """Get full metadata for a gym."""
        return self.gym_metadata.get(gym_id)

    def get_feature_slice(self, feature_name: str) -> Tuple[int, int]:
        return self.feature_dimensions.get(feature_name, (0, 0))


class UserProfileBuilder:
    """Builds user profiles from behavior and preferences."""

    def __init__(self, users_df: pd.DataFrame, checkins_df: pd.DataFrame,
                 gym_feature_builder: GymFeatureBuilder,
                 feature_extractor: FeatureExtractor,
                 recency_decay_days: int = 30):
        self.users_df = users_df
        self.checkins_df = checkins_df
        self.gym_feature_builder = gym_feature_builder
        self.feature_extractor = feature_extractor
        self.recency_decay_days = recency_decay_days

        if 'checkin_time' in self.checkins_df.columns:
            self.checkins_df['checkin_time'] = pd.to_datetime(
                self.checkins_df['checkin_time'],
                errors='coerce'
            )

        self.user_profiles = {}
        self.user_checkin_stats = {}
        self.user_behavior_patterns = {}

    def build_profile(self, user_id: str) -> Tuple[np.ndarray, Dict]:
        user_row = self.users_df[self.users_df['user_id'] == user_id]
        if user_row.empty:
            raise ValueError(f"User {user_id} not found")

        user = user_row.iloc[0]
        user_checkins = self.checkins_df[self.checkins_df['user_id'] == user_id].copy()
        checkin_count = len(user_checkins)

        # Extract behavior patterns
        behavior_patterns = self._extract_behavior_patterns(user_checkins)
        self.user_behavior_patterns[user_id] = behavior_patterns

        metadata = self._compute_user_metadata(user, user_checkins, checkin_count, behavior_patterns)

        if metadata['tier'] == 'cold_start':
            profile_vector = self._build_cold_start_profile(user)
        elif metadata['tier'] == 'warming':
            profile_vector = self._build_warming_profile(user, user_checkins, metadata)
        else:
            profile_vector = self._build_warm_profile(user, user_checkins, metadata)

        self.user_profiles[user_id] = profile_vector
        self.user_checkin_stats[user_id] = metadata

        return profile_vector, metadata

    def _extract_behavior_patterns(self, checkins: pd.DataFrame) -> Dict:
        """Extract detailed behavior patterns for explanations."""
        patterns = {
            'visited_gyms': [],
            'gym_type_preferences': Counter(),
            'facility_usage': Counter(),
            'city_preferences': Counter(),
            'most_recent_gyms': [],
            'consistent_gyms': []
        }

        if checkins.empty:
            return patterns

        # Get gym details for each checkin
        for _, checkin in checkins.iterrows():
            gym_id = checkin['gym_id']
            gym_info = self.gym_feature_builder.gyms_df[
                self.gym_feature_builder.gyms_df['gym_id'] == gym_id
            ]
            
            if not gym_info.empty:
                gym = gym_info.iloc[0]
                patterns['visited_gyms'].append(gym_id)
                patterns['gym_type_preferences'][gym.get('gym_type', 'Standard')] += 1
                patterns['city_preferences'][gym.get('city', 'Unknown')] += 1
                
                # Track facilities
                facilities = gym.get('facilities', '')
                if pd.notna(facilities):
                    for fac in facilities.split(','):
                        patterns['facility_usage'][fac.strip()] += 1

        # Most recent gyms (last 5)
        if 'checkin_time' in checkins.columns:
            recent = checkins.nlargest(5, 'checkin_time')
            patterns['most_recent_gyms'] = recent['gym_id'].tolist()

        # Consistent gyms (visited 3+ times)
        gym_counts = Counter(patterns['visited_gyms'])
        patterns['consistent_gyms'] = [gym for gym, count in gym_counts.items() if count >= 3]

        return patterns

    def _compute_user_metadata(self, user: pd.Series, checkins: pd.DataFrame,
                               checkin_count: int, behavior_patterns: Dict) -> Dict:
        metadata = {
            'checkin_count': checkin_count,
            'is_cold_start': checkin_count < 3,
            'tier': 'cold_start',
            'recency_score': 0.0,
            'diversity_score': 0.0,
            'behavior_patterns': behavior_patterns
        }

        if checkin_count == 0:
            metadata['tier'] = 'cold_start'
        elif checkin_count <= 2:
            metadata['tier'] = 'cold_start'
        elif checkin_count <= 9:
            metadata['tier'] = 'warming'
        elif checkin_count <= 29:
            metadata['tier'] = 'warm'
        else:
            metadata['tier'] = 'hot'

        if checkin_count > 0:
            now = pd.Timestamp.now()
            latest_checkin = checkins['checkin_time'].max()
            if pd.notna(latest_checkin):
                days_since = (now - latest_checkin).days
                metadata['recency_score'] = np.exp(-days_since / self.recency_decay_days)

            unique_gyms = checkins['gym_id'].nunique()
            metadata['diversity_score'] = min(unique_gyms / 10.0, 1.0)

        return metadata

    def _build_cold_start_profile(self, user: pd.Series) -> np.ndarray:
        total_dims = len(next(iter(self.gym_feature_builder.gym_features.values())))
        profile = np.zeros(total_dims)

        # Map activity to gym type
        type_start, type_end = self.gym_feature_builder.get_feature_slice('gym_type')
        if type_start < type_end:
            activity_pref = user.get('activity_preference', 'Standard')
            gym_type_mapping = {
                'Weightlifting': 'Premium',
                'Cardio': 'Standard',
                'Yoga': 'Standard',
                'CrossFit': 'Premium',
                'General': 'Standard'
            }
            mapped_type = gym_type_mapping.get(str(activity_pref), 'Standard')
            type_vector = self.feature_extractor.encode_gym_type(mapped_type)
            profile[type_start:type_end] = type_vector

        # Facilities - neutral
        facilities_start, facilities_end = self.gym_feature_builder.get_feature_slice('facilities')
        if facilities_start < facilities_end:
            profile[facilities_start:facilities_end] = 0.3

        # Price preference
        price_start, price_end = self.gym_feature_builder.get_feature_slice('price')
        if price_start < price_end:
            subscription = user.get('subscription_plan', 'Basic')
            price_preference_map = {
                'Student': 0.2,
                'DayPass': 0.3,
                'Basic': 0.5,
                'Pro': 0.7,
                'Elite': 0.9
            }
            price_pref = price_preference_map.get(str(subscription), 0.5)
            profile[price_start] = price_pref

        return profile

    def _build_warming_profile(self, user: pd.Series, checkins: pd.DataFrame,
                               metadata: Dict) -> np.ndarray:
        profile = self._build_cold_start_profile(user)
        weighted_gym_features = self._compute_weighted_gym_features(checkins)

        if weighted_gym_features is not None:
            profile = 0.6 * profile + 0.4 * weighted_gym_features

        return profile

    def _build_warm_profile(self, user: pd.Series, checkins: pd.DataFrame,
                           metadata: Dict) -> np.ndarray:
        behavioral_profile = self._compute_weighted_gym_features(checkins)

        if behavioral_profile is None:
            return self._build_cold_start_profile(user)

        explicit_profile = self._build_cold_start_profile(user)
        profile = 0.8 * behavioral_profile + 0.2 * explicit_profile

        if metadata['diversity_score'] > 0.6:
            smoothing_factor = 0.85
            profile = smoothing_factor * profile + (1 - smoothing_factor) * np.mean(profile)

        return profile

    def _compute_weighted_gym_features(self, checkins: pd.DataFrame) -> Optional[np.ndarray]:
        if checkins.empty:
            return None

        now = pd.Timestamp.now()
        total_dims = len(next(iter(self.gym_feature_builder.gym_features.values())))
        weighted_sum = np.zeros(total_dims)
        total_weight = 0.0

        for _, checkin in checkins.iterrows():
            gym_id = checkin['gym_id']
            gym_features = self.gym_feature_builder.get_feature_vector(gym_id)

            if gym_features is None:
                continue

            if 'checkin_time' in checkin and pd.notna(checkin['checkin_time']):
                days_ago = (now - checkin['checkin_time']).days
                weight = np.exp(-days_ago / self.recency_decay_days)
            else:
                weight = 1.0

            weighted_sum += weight * gym_features
            total_weight += weight

        if total_weight == 0:
            return None

        return weighted_sum / total_weight

    def get_profile(self, user_id: str) -> Optional[np.ndarray]:
        if user_id not in self.user_profiles:
            profile, _ = self.build_profile(user_id)
            return profile
        return self.user_profiles[user_id]

    def get_user_stats(self, user_id: str) -> Optional[Dict]:
        return self.user_checkin_stats.get(user_id)

    def get_behavior_patterns(self, user_id: str) -> Optional[Dict]:
        return self.user_behavior_patterns.get(user_id)


class SimilarityEngine:
    """Computes multi-dimensional similarity."""

    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        self.feature_weights = feature_weights or {
            'facilities': 0.40,
            'gym_type': 0.35,
            'price': 0.25,
        }

        total = sum(self.feature_weights.values())
        self.feature_weights = {k: v/total for k, v in self.feature_weights.items()}

    def compute_similarity(self, user_vector: np.ndarray, gym_vector: np.ndarray,
                          feature_dimensions: Dict[str, Tuple[int, int]]) -> Tuple[float, Dict[str, float]]:
        feature_scores = {}
        overall_score = 0.0

        for feature_name, (start, end) in feature_dimensions.items():
            if feature_name not in self.feature_weights:
                continue

            user_slice = user_vector[start:end]
            gym_slice = gym_vector[start:end]

            if feature_name in ['facilities']:
                sim = self._cosine_similarity(user_slice, gym_slice)
            elif feature_name in ['gym_type']:
                sim = self._binary_similarity(user_slice, gym_slice)
            elif feature_name == 'price':
                sim = 1.0 - abs(user_slice[0] - gym_slice[0])
            else:
                sim = self._cosine_similarity(user_slice, gym_slice)

            feature_scores[feature_name] = sim
            overall_score += self.feature_weights[feature_name] * sim

        return overall_score, feature_scores

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return (similarity + 1.0) / 2.0

    def _binary_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0

        intersection = np.sum(np.minimum(vec1, vec2))
        union = np.sum(np.maximum(vec1, vec2))

        if union == 0:
            return 0.0

        return intersection / union


class RecommendationRanker:
    """Ranks gyms with popularity and diversity."""

    def __init__(self, gyms_df: pd.DataFrame, checkins_df: pd.DataFrame):
        self.gyms_df = gyms_df
        self.checkins_df = checkins_df
        self.gym_popularity = self._compute_gym_popularity()
        self.gym_diversity_groups = self._compute_diversity_groups()

    def _compute_gym_popularity(self) -> Dict[str, float]:
        checkin_counts = self.checkins_df['gym_id'].value_counts()

        if len(checkin_counts) == 0:
            return {}

        max_count = checkin_counts.max()
        popularity = {}

        for gym_id in self.gyms_df['gym_id']:
            count = checkin_counts.get(gym_id, 0)
            popularity[gym_id] = count / max_count if max_count > 0 else 0.0

        return popularity

    def _compute_diversity_groups(self) -> Dict[str, str]:
        diversity_groups = {}

        for _, gym in self.gyms_df.iterrows():
            gym_id = gym['gym_id']
            gym_type = gym.get('gym_type', 'Standard')
            city = gym.get('city', 'unknown')
            group = f"{gym_type}_{city}"
            diversity_groups[gym_id] = group

        return diversity_groups

    def rank_gyms(self, user_id: str, similarity_scores: Dict[str, Tuple[float, Dict]],
                  user_city: str, top_k: int = 5,
                  diversity_factor: float = 0.3) -> List[Dict]:

        candidates = []

        for gym_id, (sim_score, feature_scores) in similarity_scores.items():
            gym = self.gyms_df[self.gyms_df['gym_id'] == gym_id].iloc[0]

            gym_city = gym.get('city', '')
            city_bonus = 0.1 if (user_city and gym_city == user_city) else 0.0

            popularity_bonus = self.gym_popularity.get(gym_id, 0.0) * 0.05
            final_score = sim_score + popularity_bonus + city_bonus

            candidates.append({
                'gym_id': gym_id,
                'gym': gym,
                'similarity_score': sim_score,
                'feature_scores': feature_scores,
                'popularity_bonus': popularity_bonus,
                'city_bonus': city_bonus,
                'final_score': final_score,
                'diversity_group': self.gym_diversity_groups.get(gym_id, 'unknown')
            })

        if not candidates:
            return []

        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        if diversity_factor > 0:
            selected = self._select_diverse_top_k(candidates, top_k, diversity_factor)
        else:
            selected = candidates[:top_k]

        recommendations = []
        for rank, candidate in enumerate(selected, 1):
            rec = self._build_recommendation_dict(rank, candidate)
            recommendations.append(rec)

        return recommendations

    def _select_diverse_top_k(self, candidates: List[Dict], top_k: int,
                             diversity_factor: float) -> List[Dict]:
        selected = []
        remaining = candidates.copy()
        selected_groups = set()

        while len(selected) < top_k and remaining:
            if len(selected) == 0:
                best = remaining.pop(0)
                selected.append(best)
                selected_groups.add(best['diversity_group'])
            else:
                best_idx = 0
                best_mmr = -float('inf')

                for idx, candidate in enumerate(remaining):
                    relevance = candidate['final_score']
                    diversity = 1.0
                    if candidate['diversity_group'] in selected_groups:
                        diversity = 0.5

                    mmr = (1 - diversity_factor) * relevance + diversity_factor * diversity

                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = idx

                best = remaining.pop(best_idx)
                selected.append(best)
                selected_groups.add(best['diversity_group'])

        return selected

    def _build_recommendation_dict(self, rank: int, candidate: Dict) -> Dict:
        gym = candidate['gym']
        gym_id = gym['gym_id']
        
        # Get actual base_price from the gym data
        base_price = gym.get('base_price', 50.0)

        return {
            'rank': rank,
            'gym_id': gym_id,
            'gym_name': gym.get('gym_name', 'Unknown Gym'),
            'city': gym.get('city', 'Unknown'),
            'state': gym.get('state', ''),
            'gym_type': gym.get('gym_type', 'Standard'),
            'facilities': gym.get('facilities', ''),
            'base_price': base_price,  # Include base_price
            'price_per_month': base_price,  # For compatibility
            'similarity_score': round(candidate['similarity_score'], 3),
            'popularity_bonus': round(candidate['popularity_bonus'], 3),
            'city_bonus': round(candidate['city_bonus'], 3),
            'final_score': round(candidate['final_score'], 3),
            'feature_scores': {k: round(v, 3) for k, v in candidate['feature_scores'].items()}
        }


class EnhancedExplanationGenerator:
    """Generates natural, behavior-based explanations."""

    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor

    def generate_explanation(self, recommendation: Dict, user_profile_metadata: Dict,
                            user_data: pd.Series, behavior_patterns: Dict) -> str:
        """Generate contextual explanation based on user behavior."""
        reasons = []
        gym_type = recommendation['gym_type']
        gym_name = recommendation['gym_name']
        price = recommendation.get('price_per_month', 50.0)
        
        # Analyze user's behavior patterns
        is_cold_start = user_profile_metadata['is_cold_start']
        checkin_count = user_profile_metadata['checkin_count']
        
        if not is_cold_start and behavior_patterns:
            # Use behavioral signals for warm users
            top_gym_types = behavior_patterns['gym_type_preferences'].most_common(2)
            top_facilities = behavior_patterns['facility_usage'].most_common(3)
            consistent_gyms = behavior_patterns['consistent_gyms']
            
            # Gym type match
            if top_gym_types and gym_type in [gt[0] for gt in top_gym_types]:
                reasons.append(f"matches your preference for {gym_type} gyms (you've visited {top_gym_types[0][1]} similar gyms)")
            
            # Facility match
            gym_facilities = [f.strip() for f in recommendation.get('facilities', '').split(',')]
            matched_facilities = []
            for fac, count in top_facilities:
                if fac in gym_facilities:
                    matched_facilities.append(fac)
            
            if matched_facilities:
                reasons.append(f"has {', '.join(matched_facilities[:2])} which you frequently use")
            
            # Similar to consistent gyms
            if consistent_gyms and recommendation['gym_id'] not in consistent_gyms:
                reasons.append(f"is similar to gyms you visit regularly")
            
            # Location
            if recommendation.get('city_bonus', 0) > 0:
                reasons.append(f"is in {recommendation['city']}, where you typically workout")
            
            # Popularity
            if recommendation.get('popularity_bonus', 0) > 0.02:
                reasons.append("is popular among similar users")
                
        else:
            # Cold start - use explicit preferences
            activity_pref = user_data.get('activity_preference', 'General')
            subscription = user_data.get('subscription_plan', 'Basic')
            
            reasons.append(f"aligns with your {activity_pref} activity preference")
            reasons.append(f"fits your {subscription} subscription budget at ${price:.0f}/month")
            
            feature_scores = recommendation.get('feature_scores', {})
            if feature_scores.get('facilities', 0) > 0.5:
                reasons.append("offers comprehensive facilities")

        # Limit to top 3 reasons
        reasons = reasons[:3]

        if reasons:
            explanation = f"{gym_name} is recommended because it {', and '.join(reasons)}."
        else:
            explanation = f"{gym_name} is recommended based on your profile (match score: {recommendation['similarity_score']:.2f})."

        # Add context
        if not is_cold_start:
            explanation += f" [Based on {checkin_count} previous visits]"
        else:
            explanation += " [Based on your stated preferences]"

        return explanation


class ContentBasedGymRecommender:
    """Main recommender with hybrid capability."""

    def __init__(self, gyms_df: pd.DataFrame, users_df: pd.DataFrame,
                 checkins_df: pd.DataFrame, recency_decay_days: int = 30,
                 feature_weights: Optional[Dict[str, float]] = None):
        
        logger.info("Initializing Content-Based Gym Recommender...")

        self.gyms_df = gyms_df
        self.users_df = users_df
        self.checkins_df = checkins_df

        self.feature_extractor = FeatureExtractor()
        self.gym_feature_builder = GymFeatureBuilder(gyms_df, self.feature_extractor)
        self.gym_features = self.gym_feature_builder.build_all_features()

        self.user_profile_builder = UserProfileBuilder(
            users_df, checkins_df, self.gym_feature_builder,
            self.feature_extractor, recency_decay_days
        )

        self.similarity_engine = SimilarityEngine(feature_weights)
        self.ranker = RecommendationRanker(gyms_df, checkins_df)
        self.explanation_generator = EnhancedExplanationGenerator(self.feature_extractor)

        logger.info("System ready!")

    def recommend(self, user_id: str, top_k: int = 5,
                 diversity_factor: float = 0.3, filter_city=None) -> List[Dict]:


        """Generate content-based recommendations."""
        
        user_data = self.users_df[self.users_df['user_id'] == user_id]
        if user_data.empty:
            raise ValueError(f"User {user_id} not found")

        user = user_data.iloc[0]

        # Build user profile
        user_profile, user_metadata = self.user_profile_builder.build_profile(user_id)
        behavior_patterns = self.user_profile_builder.get_behavior_patterns(user_id) or {}

        # Compute similarities
        similarity_scores = {}
        for gym_id, gym_features in self.gym_features.items():
            overall_sim, feature_sims = self.similarity_engine.compute_similarity(
                user_profile, gym_features,
                self.gym_feature_builder.feature_dimensions
            )
            similarity_scores[gym_id] = (overall_sim, feature_sims)

        # Rank
        user_city = user.get('city', '')
        recommendations = self.ranker.rank_gyms(
            user_id=user_id,
            similarity_scores=similarity_scores,
            user_city=user_city,
            top_k=top_k,
            diversity_factor=diversity_factor
        )

        # Generate explanations
        for rec in recommendations:
            # Add actual price from gym data
            gym_data = self.gyms_df[self.gyms_df['gym_id'] == rec['gym_id']]
            if not gym_data.empty:
                rec['price_per_month'] = gym_data.iloc[0].get('base_price', 50.0)
            else:
                rec['price_per_month'] = 50.0
            
            explanation = self.explanation_generator.generate_explanation(
                rec, user_metadata, user, behavior_patterns
            )
            rec['explanation'] = explanation

        return recommendations

    def get_user_insights(self, user_id: str) -> Dict:
        """Get user insights."""
        user_stats = self.user_profile_builder.get_user_stats(user_id)

        if user_stats is None:
            _, user_stats = self.user_profile_builder.build_profile(user_id)

        behavior_patterns = user_stats.get('behavior_patterns', {})

        insights = {
            'user_id': user_id,
            'tier': user_stats['tier'],
            'checkin_count': user_stats['checkin_count'],
            'is_cold_start': user_stats['is_cold_start'],
            'recency_score': round(user_stats['recency_score'], 3),
            'diversity_score': round(user_stats['diversity_score'], 3),
            'top_gym_types': dict(behavior_patterns.get('gym_type_preferences', {}).most_common(3)),
            'top_facilities': dict(behavior_patterns.get('facility_usage', {}).most_common(5)),
            'favorite_cities': dict(behavior_patterns.get('city_preferences', {}).most_common(3))
        }

        return insights


# ======================================================================
# EXAMPLE USAGE
# ======================================================================

if __name__ == "__main__":
    # Load data
    gyms = pd.read_csv("final-final/gyms_cleaned.csv")
    users = pd.read_csv("final-final/users_cleaned.csv")
    checkins = pd.read_csv("final-final/checkin_checkout_history_expanded.csv")
    gym_names = pd.read_csv("final-final/gym_names.csv")

    # Merge gym names
    gyms = gyms.merge(gym_names[['gym_id', 'gym_name']], on='gym_id', how='left')
    gyms['gym_name'] = gyms['gym_name'].fillna("Unnamed Gym")

    print(f"Loaded: {gyms.shape[0]} gyms, {users.shape[0]} users, {checkins.shape[0]} checkins")

    # Initialize recommender
    recommender = ContentBasedGymRecommender(
        gyms_df=gyms,
        users_df=users,
        checkins_df=checkins,
        recency_decay_days=30
    )

    # Get recommendations
    user_id = users['user_id'].iloc[0]
    print(f"\nGenerating recommendations for: {user_id}")

    recommendations = recommender.recommend(
        user_id=user_id,
        top_k=5,
        diversity_factor=0.3
    )

    # Display recommendations
    print("\n" + "="*80)
    print("TOP RECOMMENDATIONS")
    print("="*80)
    for rec in recommendations:
        print(f"\n{rec['rank']}. {rec['gym_name']}")
        print(f"   Location: {rec['city']}, {rec['state']}")
        print(f"   Type: {rec['gym_type']}")
        print(f"   Price: ${rec['price_per_month']:.2f}/month")
        print(f"   Similarity Score: {rec['similarity_score']:.3f}")
        print(f"   Final Score: {rec['final_score']:.3f}")
        print(f"   Explanation: {rec['explanation']}")

    # User insights
    insights = recommender.get_user_insights(user_id)
    print("\n" + "="*80)
    print("USER INSIGHTS")
    print("="*80)
    for key, value in insights.items():
        print(f"  {key}: {value}")