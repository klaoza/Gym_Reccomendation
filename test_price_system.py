"""
Test script for the updated gym recommendation system with price-based filtering
"""
import pandas as pd

# Simulate the price mapping
price_map = {
    'Budget': 15.0,      # Under $20
    'Standard': 32.5,    # Between $20-45
    'Premium': 55.0      # More than $45
}

def apply_student_discount(price, age):
    """Apply $10 discount for users under 18"""
    if age < 18:
        return max(0, price - 10.0)
    return price

def test_price_scenarios():
    """Test different price scenarios"""
    print("=" * 60)
    print("TESTING PRICE SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        {"name": "Adult, Budget $30", "age": 25, "max_price": 30},
        {"name": "Student, Budget $30", "age": 17, "max_price": 30},
        {"name": "Student, Budget $50", "age": 16, "max_price": 50},
        {"name": "Adult, Budget $50", "age": 30, "max_price": 50},
        {"name": "Young Student, Budget $20", "age": 15, "max_price": 20},
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"Age: {scenario['age']}, Max Price: ${scenario['max_price']}")
        print("-" * 40)
        
        for gym_type, original_price in price_map.items():
            final_price = apply_student_discount(original_price, scenario['age'])
            within_budget = final_price <= scenario['max_price']
            
            status = "✓ INCLUDED" if within_budget else "✗ EXCLUDED"
            discount_info = f" (${original_price} - $10 discount)" if scenario['age'] < 18 else ""
            
            print(f"{gym_type:10} - ${final_price:5.2f}/mo{discount_info:30} {status}")

def test_utility_scoring():
    """Test utility score calculation"""
    print("\n" + "=" * 60)
    print("TESTING UTILITY SCORE CALCULATION")
    print("=" * 60)
    
    # Simulate a gym
    gym = {
        'gym_type': 'Standard',
        'price_per_month': 32.5,
        'facilities': ['Swimming Pool', 'Personal Training Area', 'Spa & Recovery Zone'],
        'state': 'California'
    }
    
    # User preferences
    preferences = {
        'age': 17,
        'max_price': 30,
        'desired_facilities': ['Swimming Pool', 'Personal Training Area'],
        'preferred_state': 'California'
    }
    
    # Weights
    weights = {
        'price': 0.3,
        'facilities': 0.3,
        'gender': 0.2,
        'state': 0.2
    }
    
    print(f"\nGym: {gym['gym_type']} - ${gym['price_per_month']}/mo")
    print(f"User: Age {preferences['age']}, Max Budget ${preferences['max_price']}")
    print(f"Desired Facilities: {preferences['desired_facilities']}")
    print("-" * 40)
    
    # Calculate scores
    final_price = apply_student_discount(gym['price_per_month'], preferences['age'])
    print(f"\n1. Price Score:")
    print(f"   Original: ${gym['price_per_month']}")
    print(f"   After discount: ${final_price}")
    print(f"   Max budget: ${preferences['max_price']}")
    
    if final_price <= preferences['max_price']:
        price_score = 1.0
        print(f"   Score: {price_score} (within budget!)")
    else:
        price_score = max(0.0, 1.0 - (final_price - preferences['max_price']) / preferences['max_price'])
        print(f"   Score: {price_score:.2f} (over budget by ${final_price - preferences['max_price']:.2f})")
    
    # Facilities score
    matched_facilities = [f for f in preferences['desired_facilities'] if f in gym['facilities']]
    facilities_score = len(matched_facilities) / len(preferences['desired_facilities'])
    print(f"\n2. Facilities Score:")
    print(f"   Matched: {matched_facilities}")
    print(f"   Score: {facilities_score:.2f} ({len(matched_facilities)}/{len(preferences['desired_facilities'])} matched)")
    
    # Gender score (assume compatible)
    gender_score = 1.0
    print(f"\n3. Gender Score: {gender_score} (compatible)")
    
    # State score
    state_score = 1.0 if gym['state'] == preferences['preferred_state'] else 0.5
    print(f"\n4. State Score: {state_score} ({'same state' if state_score == 1.0 else 'different state'})")
    
    # Total utility
    utility = (
        weights['price'] * price_score +
        weights['facilities'] * facilities_score +
        weights['gender'] * gender_score +
        weights['state'] * state_score
    )
    
    print(f"\n{'='*40}")
    print(f"TOTAL UTILITY SCORE: {utility:.3f} ({int(utility*100)}%)")
    print(f"{'='*40}")
    print(f"Breakdown:")
    print(f"  Price      ({weights['price']:.1f}): {weights['price'] * price_score:.3f}")
    print(f"  Facilities ({weights['facilities']:.1f}): {weights['facilities'] * facilities_score:.3f}")
    print(f"  Gender     ({weights['gender']:.1f}): {weights['gender'] * gender_score:.3f}")
    print(f"  State      ({weights['state']:.1f}): {weights['state'] * state_score:.3f}")

def test_price_ranges():
    """Show which gym types are included at different price points"""
    print("\n" + "=" * 60)
    print("GYM TYPES INCLUDED BY PRICE RANGE")
    print("=" * 60)
    
    print("\nFor ADULTS (no discount):")
    adult_ranges = [10, 20, 30, 40, 50, 60]
    for max_price in adult_ranges:
        included = []
        for gym_type, price in price_map.items():
            if price <= max_price:
                included.append(gym_type)
        print(f"  Max ${max_price}: {', '.join(included) if included else 'None'}")
    
    print("\nFor STUDENTS (under 18, with $10 discount):")
    student_ranges = [10, 20, 30, 40, 50, 60]
    for max_price in student_ranges:
        included = []
        for gym_type, price in price_map.items():
            discounted_price = apply_student_discount(price, 17)
            if discounted_price <= max_price:
                included.append(f"{gym_type} (${discounted_price:.0f})")
        print(f"  Max ${max_price}: {', '.join(included) if included else 'None'}")

if __name__ == "__main__":
    test_price_scenarios()
    test_utility_scoring()
    test_price_ranges()
    
    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
