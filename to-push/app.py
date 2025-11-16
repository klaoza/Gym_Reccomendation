import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from knowledge_based import GymRecommender

# Configure page
st.set_page_config(
    page_title="Gym Recommendation System",
    page_icon="🏋️",
    layout="wide"
)

# Initialize session state
if 'recommender' not in st.session_state:
    try:
        st.session_state.recommender = GymRecommender(
            'final-final/gym_locations_expanded.csv',
            'final-final/gym_names.csv'
        )
    except Exception as e:
        st.error("Failed to load gym data. Please make sure the data file exists.")
        st.stop()

recommender = st.session_state.recommender

# Header
st.title("Gym Recommendation System")
st.markdown("### Find your perfect gym based on your preferences")
st.divider()

# Sidebar for preferences
st.sidebar.header(" Your Preferences")

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
st.sidebar.markdown("### 💰 Budget Range")
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
if adjusted_max >= 5:
    if 15.0 <= adjusted_max:
        included_types.append("Budget (<$20)")
    if 20.0 <= adjusted_max:
        included_types.append("Standard ($20-45)")
    if 45.0 < adjusted_max:
        included_types.append("Premium (>$45)")

if included_types:
    st.sidebar.info(f" Types included: {', '.join(included_types)}")

st.sidebar.divider()

# State selection
state = st.sidebar.selectbox(
    "Select State",
    options=recommender.get_available_states(),
    help="Choose the state where you want to find a gym"
)

# City selection (filtered by state)
cities = recommender.get_available_cities(state=state)
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
st.sidebar.markdown("### Desired Facilities")
all_facilities = recommender.get_all_facilities()
desired_facilities = st.sidebar.multiselect(
    "Select facilities you want",
    options=all_facilities,
    default=[],
    help="Choose the facilities that are important to you"
)

# Weight adjustments (advanced)
with st.sidebar.expander("Advanced Settings"):
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
        # Normalize weights
        price_weight /= total_weight
        facilities_weight /= total_weight
        gender_weight /= total_weight
        city_weight /= total_weight
    
    st.info(f"✓ Total weight: {price_weight + facilities_weight + gender_weight + city_weight:.2f}")

st.sidebar.divider()

# Recommendation button
if st.sidebar.button("Find Gyms", type="primary", use_container_width=True):
    if max_price < min_price:
        st.error("Maximum price must be greater than minimum price!")
    elif not desired_facilities:
        st.warning("Please select at least one facility for better recommendations.")
    else:
        with st.spinner("Finding the best gyms for you..."):
            # Get recommendations
            weights = {
                "price": price_weight,
                "facilities": facilities_weight,
                "gender": gender_weight,
                "city": city_weight
            }
            
            recommendations = recommender.recommend_with_utility_function(
                city=city,
                gender=gender,
                min_price=min_price,
                max_price=max_price,
                age=age,
                desired_facilities=desired_facilities,
                preferred_city=city,  # Use the selected city as preferred
                weights=weights
            )
            
            # Store in session state
            st.session_state.recommendations = recommendations
            st.session_state.preferences = {
                "city": city,
                "state": state,
                "gender": gender,
                "min_price": min_price,
                "max_price": max_price,
                "age": age,
                "desired_facilities": desired_facilities
            }

# Display results
if 'recommendations' in st.session_state and not st.session_state.recommendations.empty:
    recommendations = st.session_state.recommendations
    prefs = st.session_state.preferences
    
    # Summary
    st.success(f"Found {len(recommendations)} gyms in {prefs['city']}, {prefs['state']}")
    
    # Display preferences summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Location", f"{prefs['city']}, {prefs['state']}")
    with col2:
        st.metric("Age", f"{prefs['age']} years")
    with col3:
        discount_text = " (with discount)" if prefs['age'] < 18 else ""
        st.metric("Budget", f"${prefs['min_price']:.0f}-${prefs['max_price']:.0f}{discount_text}")
    with col4:
        st.metric("Facilities Selected", len(prefs['desired_facilities']))
    
    st.divider()
    
    # Top 3 Recommendations
    st.markdown("##Top Recommended Gyms")
    
    top_3 = recommendations.head()
    
    for idx, (_, gym) in enumerate(top_3.iterrows(), 1):
        # Medal emoji
        medal = "" 
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                gym_display_name = gym.get('gym_name', gym['gym_id'].upper())
                st.markdown(f"### {medal} {idx}. {gym_display_name}")
                
                # Price and type info
                final_price = gym['final_price']
                original_price = gym['price_per_month']
                
                if prefs['age'] < 18 and original_price != final_price:
                    st.markdown(f"**Type:** {gym['gym_type']} | **Price:** ~~${original_price:.2f}~~ **${final_price:.2f}/month** 🎓")
                else:
                    st.markdown(f"**Type:** {gym['gym_type']} | **Price:** ${final_price:.2f}/month")
                
                st.markdown(f"**Gender:** {', '.join(gym['gender'])}")
                
                # Facilities
                gym_facilities = [fac.strip() for fac in gym['facilities'].split(',')]
                st.markdown(f"**Facilities:** {gym['facilities']}")
                
                # Matched facilities
                if prefs['desired_facilities']:
                    matched = [fac for fac in prefs['desired_facilities'] if fac in gym_facilities]
                    if matched:
                        st.markdown(f"✅ **Matched facilities:** {', '.join(matched)}")
                    unmatched = [fac for fac in prefs['desired_facilities'] if fac not in gym_facilities]
                    if unmatched:
                        st.markdown(f"❌ **Missing facilities:** {', '.join(unmatched)}")
            
            with col2:
                # Score display
                score = gym['utility_score']
                score_percentage = int(score * 100)
                
                st.metric("Match Score", f"{score_percentage}%")
                
                # Progress bar
                st.progress(score)
                
                # Price indicator
                final_price = gym['final_price']
                if final_price <= prefs['max_price']:
                    st.success(f"✓ ${final_price:.0f}/mo")
                else:
                    st.warning(f"⚠ ${final_price:.0f}/mo")
            
            st.divider()
    
    # All recommendations table
    if len(recommendations) > 3:
        with st.expander(f"📋 View All {len(recommendations)} Recommendations"):
            # Select columns, prioritizing gym_name if available
            columns_to_show = ['gym_type', 'final_price', 'facilities', 'utility_score']
            if 'gym_name' in recommendations.columns:
                columns_to_show = ['gym_name'] + columns_to_show
            else:
                columns_to_show = ['gym_id'] + columns_to_show
            
            display_df = recommendations[columns_to_show].copy()
            display_df['utility_score'] = display_df['utility_score'].round(2)
            display_df['match_percentage'] = (display_df['utility_score'] * 100).round(0).astype(int).astype(str) + '%'
            display_df['final_price'] = '$' + display_df['final_price'].round(2).astype(str) + '/mo'
            
            rename_dict = {
                'gym_type': 'Type',
                'final_price': 'Price',
                'facilities': 'Facilities',
                'utility_score': 'Score',
                'match_percentage': 'Match %'
            }
            if 'gym_name' in display_df.columns:
                rename_dict['gym_name'] = 'Gym Name'
            else:
                rename_dict['gym_id'] = 'Gym ID'
            
            display_df = display_df.rename(columns=rename_dict)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

elif 'recommendations' in st.session_state and st.session_state.recommendations.empty:
    st.warning(f"⚠️ No gyms found in {st.session_state.preferences['city']} matching your criteria.")
    st.info("Try selecting a different city or adjusting your preferences.")

else:
    # Welcome screen
    st.info("Please set your preferences in the sidebar and click 'Find Gyms' to get recommendations.")
    
    # Display some statistics
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Gyms", len(recommender.gyms_df))
    with col2:
        st.metric("States", recommender.gyms_df['state'].nunique())
    with col3:
        st.metric("Cities", recommender.gyms_df['city'].nunique())
    with col4:
        st.metric("Facilities", len(recommender.get_all_facilities()))
    
    # Show gym distribution by type
    st.markdown("### Gym Types Distribution & Pricing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gym_type_dist = recommender.gyms_df['gym_type'].value_counts()
        st.bar_chart(gym_type_dist)
    
    with col2:
        st.markdown("**Price Ranges:**")
        st.markdown("- 💚 **Budget**: <$20/month (avg $15)")
        st.markdown("- 💙 **Standard**: $20-45/month (avg $32.50)")
        st.markdown("- 💎 **Premium**: >$45/month (avg $55)")
        st.markdown("")
        st.info("🎓 Students under 18 get $10 discount on all tiers!")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Powered by Knowledge-Based Recommendation System | 
        Utility Function Scoring</p>
    </div>
    """,
    unsafe_allow_html=True
)
