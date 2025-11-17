import pandas as pd
from typing import List, Dict, Any

class GymRecommender:
    """
    Knowledge-based gym recommendation system using utility function.
    Now uses actual base_price from dataset instead of fixed price map.
    """
    def __init__(self, gyms_data_path: str, gym_names_path: str = None):
        """
        Initialize the recommender by loading gym data.
        """
        try:
            self.gyms_df = pd.read_csv(gyms_data_path)
            
            # Preprocess gender column to convert to list
            self.gyms_df['gender'] = self.gyms_df['gender'].str.split(', ')
            
            # Load gym names if provided
            if gym_names_path:
                gym_names_df = pd.read_csv(gym_names_path)
                self.gyms_df = self.gyms_df.merge(gym_names_df, on='gym_id', how='left')
            
            # Use actual base_price from dataset
            if 'base_price' not in self.gyms_df.columns:
                raise ValueError("Dataset must contain 'base_price' column")
                
            # Ensure base_price is numeric
            self.gyms_df['base_price'] = pd.to_numeric(self.gyms_df['base_price'], errors='coerce')
            
            # Fill any missing prices with median
            median_price = self.gyms_df['base_price'].median()
            self.gyms_df['base_price'] = self.gyms_df['base_price'].fillna(median_price)
            
            print(f"Loaded {len(self.gyms_df)} gyms")
            print(f"Price range: ${self.gyms_df['base_price'].min():.2f} - ${self.gyms_df['base_price'].max():.2f}")
            print(f"Average price: ${self.gyms_df['base_price'].mean():.2f}")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}. Please check the file path.")
            raise

    def _calculate_utility_score(self, gym: pd.Series, preferences: Dict[str, Any], weights: Dict[str, float]) -> float:
        """
        Calculate utility score for a single gym based on preferences and weights.
        U(i) = Σ wa * scorea(i[a])
        """
        total_score = 0.0
        
        # Score for price (within budget after age discount)
        age = preferences.get('age', 25)
        max_price = preferences.get('max_price', 100)
        gym_price = gym['base_price']  
        
        # Apply student discount if under 18
        if age < 18:
            gym_price = max(0, gym_price - 10.0)
        
        # Price score: 1.0 if within budget, decreases proportionally if over budget
        if gym_price <= max_price:
            score_price = 1.0
        else:
            # Penalize gyms over budget
            score_price = max(0.0, 1.0 - (gym_price - max_price) / max_price)
        
        # Score for facilities
        desired_facilities = preferences.get('desired_facilities', [])
        score_facilities = 0.0
        if desired_facilities:
            gym_facilities_str = gym.get('facilities', '')
            if pd.isna(gym_facilities_str):
                gym_facilities_list = []
            else:
                gym_facilities_list = [fac.strip() for fac in gym_facilities_str.split(',')]
            
            matched_facilities = sum(1 for fac in desired_facilities if fac in gym_facilities_list)
            score_facilities = matched_facilities / len(desired_facilities) if desired_facilities else 0.0
        
        # Score for gender preference match
        user_gender = preferences.get('gender', '')
        score_gender = 1.0  # Default: compatible
        
        # Score for city preference (if user prefers certain city)
        preferred_city = preferences.get('preferred_city', None)
        score_city = 1.0 if (preferred_city is None or gym.get('city') == preferred_city) else 0.5
        
        # Calculate additive utility
        utility_score = (
            weights.get('price', 0.3) * score_price +
            weights.get('facilities', 0.3) * score_facilities +
            weights.get('gender', 0.1) * score_gender +
            weights.get('city', 0.1) * score_city
        )
        
        return utility_score

    def recommend_with_utility_function(
        self, 
        city: str, 
        gender: str, 
        min_price: float,
        max_price: float,
        age: int,
        desired_facilities: List[str],
        preferred_city: str = None,
        weights: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Recommend gyms using utility function to score and rank.
        """
        # Step 1: Initial filtering (strict constraints)
        candidate_gyms = self.gyms_df[self.gyms_df['city'].str.lower() == city.lower()].copy()

        # Filter by gender compatibility
        def is_gender_compatible(gym_genders):
            if gender == 'Non-binary':
                return 'Male' in gym_genders or 'Female' in gym_genders
            return gender in gym_genders

        candidate_gyms = candidate_gyms[candidate_gyms['gender'].apply(is_gender_compatible)]

        # Filter by price range (considering student discount)
        def is_price_acceptable(gym_price):
            discounted_price = gym_price - 10.0 if age < 18 else gym_price
            # Include gyms within max price OR close to min price
            return discounted_price <= max_price
        
        candidate_gyms = candidate_gyms[candidate_gyms['base_price'].apply(is_price_acceptable)]

        if candidate_gyms.empty:
            return pd.DataFrame()

        # Step 2: Calculate utility score
        preferences = {
            "age": age,
            "max_price": max_price,
            "min_price": min_price,
            "desired_facilities": desired_facilities,
            "gender": gender,
            "preferred_city": preferred_city
        }
        
        # Default weights if not provided
        if weights is None:
            weights = {
                "price": 0.3,
                "facilities": 0.3,
                "gender": 0.2,
                "city": 0.2
            }

        # Apply utility function to each candidate gym
        candidate_gyms['utility_score'] = candidate_gyms.apply(
            self._calculate_utility_score, 
            axis=1, 
            preferences=preferences, 
            weights=weights
        )
        
        # Add final price after discount for display
        candidate_gyms['final_price'] = candidate_gyms['base_price'].apply(
            lambda p: p - 10.0 if age < 18 else p
        )
        
        # Add price_per_month for compatibility
        candidate_gyms['price_per_month'] = candidate_gyms['base_price']

        # Step 3: Sort and return results
        recommended_gyms = candidate_gyms.sort_values(by='utility_score', ascending=False)
        
        return recommended_gyms

    def get_available_cities(self, state: str = None) -> List[str]:
        """Get list of available cities, optionally filtered by state."""
        if state:
            cities = self.gyms_df[self.gyms_df['state'] == state]['city'].unique()
        else:
            cities = self.gyms_df['city'].unique()
        return sorted(cities.tolist())

    def get_available_states(self) -> List[str]:
        """Get list of available states."""
        return sorted(self.gyms_df['state'].unique().tolist())

    def get_all_facilities(self) -> List[str]:
        """Get list of all unique facilities."""
        all_facilities = set()
        for facilities in self.gyms_df['facilities'].dropna():
            for fac in facilities.split(','):
                all_facilities.add(fac.strip())
        return sorted(list(all_facilities))


if __name__ == "__main__":
    recommender = GymRecommender(
        'final-final/gyms_cleaned.csv',
        'final-final/gym_names.csv'
    )
    
    recommendations = recommender.recommend_with_utility_function(
        city='San Francisco',
        gender='Male',
        min_price=0,
        max_price=50,
        age=25,
        desired_facilities=['Yoga Studio', 'Swimming Pool'],
        preferred_city='San Francisco'
    )
    
    print(f"\nFound {len(recommendations)} gyms:")
    for idx, (_, gym) in enumerate(recommendations.head(5).iterrows(), 1):
        print(f"\n{idx}. {gym.get('gym_name', gym['gym_id'])}")
        print(f"   Price: ${gym['final_price']:.2f}/month")
        print(f"   Type: {gym['gym_type']}")
        print(f"   Utility Score: {gym['utility_score']:.3f}")