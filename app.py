import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from knowledge_based import GymRecommender as KBRecommender
from content_based import ContentBasedGymRecommender
from hybrid_recommender import HybridGymRecommender

# Configure page
st.set_page_config(
    page_title="Gym Recommendation System",
    page_icon="🏋️",
    layout="wide"
)

# Initialize session state
if 'recommenders_initialized' not in st.session_state:
    try:
        # Load data
        gyms = pd.read_csv('final-final/gyms_cleaned.csv')
        users = pd.read_csv('final-final/users_cleaned.csv')
        checkins = pd.read_csv('final-final/checkin_checkout_history_expanded.csv')
        gym_names = pd.read_csv('final-final/gym_names.csv')
        
        # Merge gym names
        gyms = gyms.merge(gym_names[['gym_id', 'gym_name']], on='gym_id', how='left')
        gyms['gym_name'] = gyms['gym_name'].fillna("Unnamed Gym")
        
        # Initialize all recommenders
        st.session_state.kb_recommender = KBRecommender(
            'final-final/gyms_cleaned.csv',
            'final-final/gym_names.csv'
        )
        
        st.session_state.cb_recommender = ContentBasedGymRecommender(
            gyms_df=gyms,
            users_df=users,
            checkins_df=checkins
        )
        
        st.session_state.hybrid_recommender = HybridGymRecommender(
            kb_recommender=st.session_state.kb_recommender,
            cb_recommender=st.session_state.cb_recommender,
            hybrid_strategy='cascade',
            kb_weight=0.4,
            cb_weight=0.6
        )
        
        st.session_state.users_df = users
        st.session_state.gyms_df = gyms
        st.session_state.recommenders_initialized = True
        
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# Get recommenders
kb_recommender = st.session_state.kb_recommender
cb_recommender = st.session_state.cb_recommender
hybrid_recommender = st.session_state.hybrid_recommender
users_df = st.session_state.users_df

# Header
st.title("🏋️ Advanced Gym Recommendation System")
st.markdown("### Find your perfect gym with AI-powered recommendations")
st.divider()

# Sidebar for preferences
st.sidebar.header(" Your Preferences")

# Recommendation method selection
rec_method = st.sidebar.radio(
    " Recommendation Method",
    options=["Hybrid (Recommended)", "Knowledge-Based Only", "Content-Based Only"],
    help="Choose how recommendations are generated"
)

# Show method explanation
with st.sidebar.expander(" What's the difference???"):
    st.markdown("""
    **Hybrid (Recommended)**: Combines both approaches
    - Uses rules to filter gyms
    - Uses AI to rank by your behavior
    - Best of both worlds!
    
    **Knowledge-Based**: Rule-based filtering
    - Filters by your explicit preferences
    - Good for new users
    - Fast and explainable
    
    **Content-Based**: AI personalization
    - Learns from your gym history
    - Finds similar gyms to what you like
    - Best for active users
    """)

st.sidebar.divider()

# User selection (always available)
user_id = st.sidebar.selectbox(
    "Select User Profile",
    options=users_df['user_id'].tolist(),
    help="Choose a user profile (used for CB and Hybrid, shown for reference in KB)"
)

# Show user insights (always available)
if st.sidebar.checkbox("Show my profile insights", value=False):
    insights = cb_recommender.get_user_insights(user_id)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Gym Visits", insights['checkin_count'])
        st.metric("Activity Level", insights['tier'].title())
    with col2:
        st.metric("Diversity Score", f"{insights['diversity_score']:.2f}")
        st.metric("Recency Score", f"{insights['recency_score']:.2f}")
    
    if insights['top_gym_types']:
        st.sidebar.markdown("**Your Preferences:**")
        for gym_type, count in list(insights['top_gym_types'].items())[:3]:
            st.sidebar.write(f"• {gym_type}: {count} visits")
    
    if rec_method == "Knowledge-Based Only":
        st.sidebar.info("ℹ️ KB doesn't use user history, but you can see insights above")

st.sidebar.divider()
if st.button("Evaluate System"):
    import evaluate_system
    st.write(evaluate_system.comparison)


# Age input
age = st.sidebar.number_input(
    "Your Age",
    min_value=10,
    max_value=100,
    value=25,
    step=1,
    help="Users under 18 get a $10 student discount"
)

if age < 18:
    st.sidebar.success("🎓 Student discount applied: -$10 per month!")

# Price range
st.sidebar.markdown("### Budget Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    min_price = st.number_input(
        "Min Price ($)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
        help="Minimum monthly price"
    )
with col2:
    max_price = st.number_input(
        "Max Price ($)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="Maximum monthly price"
    )

# Show what gym types are included
if age < 18:
    adjusted_max = max_price + 10
else:
    adjusted_max = max_price

included_types = []
if adjusted_max >= 15.0:
    included_types.append("Budget (<$20)")
if adjusted_max >= 20.0:
    included_types.append("Standard ($20-45)")
if adjusted_max > 45.0:
    included_types.append("Premium (>$45)")

if included_types:
    st.sidebar.info(f"Types included: {', '.join(included_types)}")

st.sidebar.divider()

# State selection
state = st.sidebar.selectbox(
    "Select State",
    options=kb_recommender.get_available_states(),
    help="Choose the state where you want to find a gym"
)

# City selection (filtered by state)
cities = kb_recommender.get_available_cities(state=state)
city = st.sidebar.selectbox(
    "Select City",
    options=cities,
    help="Choose your city"
)

# Gender selection
gender = st.sidebar.selectbox(
    "Select Gender",
    options=["Male", "Female"],
    help="Your gender preference for gym compatibility"
)

# Facilities selection
st.sidebar.markdown("###  Desired Facilities")
all_facilities = kb_recommender.get_all_facilities()
desired_facilities = st.sidebar.multiselect(
    "Select facilities you want",
    options=all_facilities,
    default=[],
    help="Choose the facilities that are important to you"
)

# Advanced settings for hybrid/KB
if rec_method in ["Hybrid (Recommended)", "Knowledge-Based Only"]:
    with st.sidebar.expander(" Advanced Settings"):
        st.markdown("**Adjust importance weights:**")
        
        price_weight = st.slider(
            "Price Importance",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="How important is staying within budget?"
        )
        
        facilities_weight = st.slider(
            "Facilities Importance",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="How important are the desired facilities?"
        )
        
        gender_weight = st.slider(
            "Gender Preference Importance",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="How important is gender compatibility?"
        )
        
        city_weight = st.slider(
            "City Preference Importance",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="How important is being in your preferred city?"
        )
        
        total_weight = price_weight + facilities_weight + gender_weight + city_weight
        
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight:.2f}. They will be normalized to 1.0")
            price_weight /= total_weight
            facilities_weight /= total_weight
            gender_weight /= total_weight
            city_weight /= total_weight
        
        st.info(f"✓ Total weight: {price_weight + facilities_weight + gender_weight + city_weight:.2f}")
        
        weights = {
            "price": price_weight,
            "facilities": facilities_weight,
            "gender": gender_weight,
            "city": city_weight
        }
else:
    weights = None

# Hybrid strategy selection
if rec_method == "Hybrid (Recommended)":
    with st.sidebar.expander(" Hybrid Strategy"):
        hybrid_strategy = st.radio(
            "Strategy",
            options=["cascade", "weighted", "switching"],
            help="""
            Cascade: KB filters → CB ranks
            Weighted: Combine both scores
            Switching: Auto-select based on user
            """
        )
        
        if hybrid_strategy != "switching":
            kb_weight = st.slider("KB Weight", 0.0, 1.0, 0.4, 0.05)
            cb_weight = 1.0 - kb_weight
            st.info(f"KB: {kb_weight:.0%}, CB: {cb_weight:.0%}")
            
            # Update hybrid recommender
            hybrid_recommender.hybrid_strategy = hybrid_strategy
            hybrid_recommender.kb_weight = kb_weight
            hybrid_recommender.cb_weight = cb_weight
        else:
            hybrid_recommender.hybrid_strategy = hybrid_strategy

st.sidebar.divider()

# Recommendation button
if st.sidebar.button(" Find Gyms", type="primary", use_container_width=True):
    if max_price < min_price:
        st.error("Maximum price must be greater than minimum price!")
    elif rec_method != "Knowledge-Based Only" and user_id is None:
        st.error("Please select a user profile!")
    else:
        with st.spinner("Finding the best gyms for you..."):
            
            # Get recommendations based on method
            if rec_method == "Knowledge-Based Only":
                recommendations = kb_recommender.recommend_with_utility_function(
                    city=city,
                    gender=gender,
                    min_price=min_price,
                    max_price=max_price,
                    age=age,
                    desired_facilities=desired_facilities,
                    preferred_city=city,
                    weights=weights
                )
                rec_type = "KB"
                
            elif rec_method == "Content-Based Only":
                # Get CB recommendations filtered by city
                cb_recs = cb_recommender.recommend(
                    user_id=user_id,
                    top_k=20,
                    diversity_factor=0.3,
                    filter_city=city  # Filter by selected city
                )
                
                # Convert to DataFrame
                if cb_recs:
                    recommendations = pd.DataFrame(cb_recs)
                    # Add missing columns for display
                    recommendations['utility_score'] = recommendations['similarity_score']
                    recommendations['price_per_month'] = recommendations['price_per_month']
                    recommendations['final_price'] = recommendations.apply(
                        lambda row: row['price_per_month'] - 10 if age < 18 else row['price_per_month'],
                        axis=1
                    )
                else:
                    recommendations = pd.DataFrame()
                
                rec_type = "CB"
                
            else:  # Hybrid
                recommendations = hybrid_recommender.recommend(
                    user_id=user_id,
                    city=city,
                    gender=gender,
                    min_price=min_price,
                    max_price=max_price,
                    age=age,
                    desired_facilities=desired_facilities,
                    preferred_city=city,
                    weights=weights,
                    top_k=20
                )
                rec_type = "Hybrid"
            
            # Store in session state
            st.session_state.recommendations = recommendations
            st.session_state.rec_type = rec_type
            st.session_state.preferences = {
                "city": city,
                "state": state,
                "gender": gender,
                "min_price": min_price,
                "max_price": max_price,
                "age": age,
                "desired_facilities": desired_facilities,
                "user_id": user_id,
                "method": rec_method
            }

# Display results
if 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
    recommendations = st.session_state.recommendations
    prefs = st.session_state.preferences
    rec_type = st.session_state.rec_type
    
    # Summary
    method_badge = {
        "KB": "Knowledge-Based",
        "CB": "Content-Based",
        "Hybrid": "Hybrid AI"
    }
    
    st.success(f"{method_badge[rec_type]} | Found {len(recommendations)} gyms in {prefs['city']}, {prefs['state']}")
    
    # Display preferences summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Location", f"{prefs['city']}, {prefs['state']}")
    with col2:
        if prefs.get('user_id'):
            st.metric("User", prefs['user_id'])
        else:
            st.metric("Age", f"{prefs['age']} years")
    with col3:
        discount_text = " (with discount)" if prefs['age'] < 18 else ""
        st.metric("Budget", f"${prefs['min_price']:.0f}-${prefs['max_price']:.0f}{discount_text}")
    with col4:
        st.metric("Facilities Selected", len(prefs['desired_facilities']))
    
    st.divider()
    
    # Top Recommendations
    st.markdown("##  Top Recommended Gyms")
    
    top_5 = recommendations.head(5)
    
    for idx, (_, gym) in enumerate(top_5.iterrows(), 1):
        medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "⭐"
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                gym_display_name = gym.get('gym_name', gym['gym_id'].upper())
                st.markdown(f"### {medal} {idx}. {gym_display_name}")
                
                # Price and type info
                final_price = gym.get('final_price', gym.get('price_per_month', 0))
                
                if rec_type == "KB":
                    original_price = gym['price_per_month']
                    if prefs['age'] < 18 and original_price != final_price:
                        st.markdown(f"**Type:** {gym['gym_type']} | **Price:** ~~${original_price:.2f}~~ **${final_price:.2f}/month** 🎓")
                    else:
                        st.markdown(f"**Type:** {gym['gym_type']} | **Price:** ${final_price:.2f}/month")
                else:
                    st.markdown(f"**Type:** {gym['gym_type']} | **Price:** ${final_price:.2f}/month")
                
                # Gender info
                if 'gender' in gym and isinstance(gym['gender'], list):
                    st.markdown(f"**Gender:** {', '.join(gym['gender'])}")
                
                # Facilities
                gym_facilities_str = gym.get('facilities', '')
                if pd.notna(gym_facilities_str):
                    gym_facilities = [fac.strip() for fac in str(gym_facilities_str).split(',')]
                    st.markdown(f"**Facilities:** {gym_facilities_str}")
                    
                    # Matched facilities
                    if prefs['desired_facilities']:
                        matched = [fac for fac in prefs['desired_facilities'] if fac in gym_facilities]
                        if matched:
                            st.markdown(f"✅ **Matched:** {', '.join(matched)}")
                        unmatched = [fac for fac in prefs['desired_facilities'] if fac not in gym_facilities]
                        if unmatched:
                            st.markdown(f"❌ **Missing:** {', '.join(unmatched)}")
                
                # Explanation
                if rec_type == "Hybrid" and 'hybrid_explanation' in gym:
                    st.info(f" {gym['hybrid_explanation']}")
                elif rec_type == "CB" and 'explanation' in gym:
                    st.info(f" {gym['explanation']}")
            
            with col2:
                # Score display based on rec type
                if rec_type == "KB":
                    score = gym['utility_score']
                    score_label = "Utility Score"
                elif rec_type == "CB":
                    score = gym.get('similarity_score', gym.get('final_score', 0))
                    score_label = "Match Score"
                else:  # Hybrid
                    score = gym.get('hybrid_score', gym.get('utility_score', 0))
                    score_label = "Hybrid Score"
                
                score_percentage = int(score * 100)
                st.metric(score_label, f"{score_percentage}%")
                st.progress(score)
                
                # Show sub-scores for hybrid
                if rec_type == "Hybrid":
                    if 'utility_score' in gym:
                        st.caption(f"KB: {gym['utility_score']:.2f}")
                    if 'cb_similarity_score' in gym:
                        st.caption(f"CB: {gym['cb_similarity_score']:.2f}")
                
                # Price indicator
                if final_price <= prefs['max_price']:
                    st.success(f"✓ ${final_price:.0f}/mo")
                else:
                    st.warning(f"⚠ ${final_price:.0f}/mo")
            
            st.divider()
    
    # All recommendations table
    if len(recommendations) > 5:
        with st.expander(f"📋 View All {len(recommendations)} Recommendations"):
            display_cols = ['gym_name', 'gym_type', 'city']
            
            # Add score column based on type
            if rec_type == "Hybrid":
                display_cols.append('hybrid_score')
            elif rec_type == "CB":
                display_cols.append('similarity_score')
            else:
                display_cols.append('utility_score')
            
            display_cols.extend(['final_price', 'facilities'])
            
            # Filter columns that exist
            available_cols = [col for col in display_cols if col in recommendations.columns]
            
            if 'final_price' not in recommendations.columns and 'price_per_month' in recommendations.columns:
                recommendations['final_price'] = recommendations['price_per_month']
            
            display_df = recommendations[available_cols].copy()
            
            # Format display
            if 'final_price' in display_df.columns:
                display_df['final_price'] = '$' + display_df['final_price'].round(2).astype(str) + '/mo'
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)

elif 'recommendations' in st.session_state and st.session_state.recommendations.empty:
    st.warning(f"⚠️ No gyms found in {st.session_state.preferences['city']} matching your criteria.")
    st.info("Try selecting a different city or adjusting your preferences.")

else:
    # Welcome screen
    st.info(f"Welcome! Set your preferences and click **'Find Gyms'** to get {rec_method.lower()} recommendations.")
    
    # Display statistics
    st.markdown("###  Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    gyms_df = st.session_state.gyms_df
    with col1:
        st.metric("Total Gyms", len(gyms_df))
    with col2:
        st.metric("States", gyms_df['state'].nunique())
    with col3:
        st.metric("Cities", gyms_df['city'].nunique())
    with col4:
        st.metric("Users", len(users_df))
    
    # Show recommendation methods comparison
    st.markdown("###  Choose Your Recommendation Style")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("####  Knowledge-Based")
        st.markdown("""
        - Rule-based filtering
        - Explicit preferences
        - Fast and transparent
        - Good for new users
        """)
    
    with col2:
        st.markdown("####  Content-Based")
        st.markdown("""
        - AI-powered learning
        - Behavior patterns
        - Personalized matches
        - Best for active users
        """)
    
    with col3:
        st.markdown("####  Hybrid (Best)")
        st.markdown("""
        - Combines both methods
        - Adaptive strategy
        - Balanced results
        - Works for everyone
        """)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Powered by Knowledge-Based, Content-Based, and Hybrid AI Systems</p>
    </div>
    """,
    unsafe_allow_html=True
)