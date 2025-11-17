import pandas as pd
import numpy as np
from datetime import datetime

# ================================================================
# LOAD CSV FILES
# ================================================================
gyms = pd.read_csv("final-final/gyms_cleaned.csv")
users = pd.read_csv("final-final/users_cleaned.csv")
checkins = pd.read_csv("final-final/checkin_checkout_history_expanded.csv")
subs = pd.read_csv("final-final/subscription_plans_clean.csv")

# ================================================================
# PREP LOOKUP TABLES
# ================================================================
user_state_map = dict(zip(users["user_id"], users["state"]))
gym_state_map = dict(zip(gyms["gym_id"], gyms["state"]))

activity_to_facility = {
    "yoga": "yoga studio",
    "swimming": "pool",
    "strength training": "free weights",
    "cardio": "cardio",
    "crossfit": "functional training",
}

def normalize(text):
    if pd.isna(text): return None
    return str(text).strip().lower()

checkins["act"] = checkins["workout_type"].apply(normalize)

def normalize_facilities(x):
    if pd.isna(x): return []
    return [f.strip().lower() for f in str(x).split(",")]

gyms["fac_list"] = gyms["facilities"].apply(normalize_facilities)


# ================================================================
# 1. CHECK FOR MISSING VALUES
# ================================================================
missing_score = 100
missing_penalty = 0

key_columns = {
    "gyms": ["gym_id", "city", "state", "base_price"],
    "users": ["user_id", "city", "state", "subscription_plan"],
    "checkins": ["user_id", "gym_id", "checkin_time", "checkout_time"],
}

for col in key_columns["gyms"]:
    if gyms[col].isna().sum() > 0:
        missing_penalty += 3

for col in key_columns["users"]:
    if users[col].isna().sum() > 0:
        missing_penalty += 3

for col in key_columns["checkins"]:
    if checkins[col].isna().sum() > 0:
        missing_penalty += 3

missing_score -= missing_penalty
missing_score = max(missing_score, 0)


# ================================================================
# 2. REFERENTIAL INTEGRITY CHECKS
# ================================================================
ref_score = 100

invalid_users = checkins[~checkins["user_id"].isin(users["user_id"])]
ref_score -= len(invalid_users) * 2

invalid_gyms = checkins[~checkins["gym_id"].isin(gyms["gym_id"])]
ref_score -= len(invalid_gyms) * 2

dupe_gyms = gyms["gym_id"].duplicated().sum()
ref_score -= dupe_gyms * 5

ref_score = max(ref_score, 0)


# ================================================================
# 3. GEOGRAPHIC CONSISTENCY
# ================================================================
geo_score = 100
geo_penalty = 0

for idx, row in checkins.iterrows():
    uid = row["user_id"]
    gid = row["gym_id"]

    user_state = user_state_map.get(uid)
    gym_state = gym_state_map.get(gid)

    if user_state is None or gym_state is None:
        continue

    if user_state != gym_state:
        geo_penalty += 5

geo_score -= geo_penalty
geo_score = max(geo_score, 0)


# ================================================================
# 4. TIME CONSISTENCY
# ================================================================
time_score = 100
time_penalty = 0

def parse_time(t):
    try:
        return datetime.fromisoformat(t)
    except:
        return None

checkins["checkin_dt"] = checkins["checkin_time"].apply(parse_time)
checkins["checkout_dt"] = checkins["checkout_time"].apply(parse_time)

for idx, r in checkins.iterrows():
    if r["checkin_dt"] is None or r["checkout_dt"] is None:
        time_penalty += 2
        continue

    if r["checkout_dt"] < r["checkin_dt"]:
        time_penalty += 5

    if (r["checkout_dt"] - r["checkin_dt"]).seconds / 3600 > 8:
        time_penalty += 3

time_score -= time_penalty
time_score = max(time_score, 0)


# ================================================================
# 5. PRICE LOGIC VALIDATION (FIXED)
# ================================================================
price_score = 100
price_penalty = 0

# 5.1 – Gym base_price must be in a realistic range
invalid_price = gyms[(gyms["base_price"] < 10) | (gyms["base_price"] > 150)]
price_penalty += len(invalid_price) * 2

# 5.2 – subscription_plans_clean must follow:
#       price_for_plan ≈ base_price * coefficient (within tolerance)
if {"base_price", "coefficient", "price_for_plan"}.issubset(subs.columns):

    expected_price = subs["base_price"] * subs["coefficient"]

    TOLERANCE = 0.01

    invalid_subs = subs[(subs["price_for_plan"] - expected_price).abs() > TOLERANCE]


    price_penalty += len(invalid_subs) * 2

else:
    print("⚠ WARNING: base_price / coefficient / price_for_plan missing in subs")
    price_penalty += 50

price_score -= price_penalty
price_score = max(price_score, 0)




# ================================================================
# FINAL SCORE COMPUTATION (WEIGHTED)
# ================================================================
final_score = (
    missing_score * 0.15 +
    ref_score * 0.20 +
    geo_score * 0.25 +
    time_score * 0.10 +
    price_score * 0.10 
)

final_score = round(final_score, 2)

if final_score >= 90:
    label = "EXCELLENT"
elif final_score >= 75:
    label = "GOOD (Minor Issues)"
elif final_score >= 50:
    label = "NEEDS FIXING"
else:
    label = "CORRUPTED DATA"


# ================================================================
# PRINT RESULTS
# ================================================================
print("========================================")
print("       DATA SANITY SCORE REPORT         ")
print("========================================")
print(f"Missing Data Score       : {missing_score}")
print(f"Referential Integrity    : {ref_score}")
print(f"Geographic Consistency   : {geo_score}")
print(f"Time Logic Score         : {time_score}")
print(f"Price Logic Score        : {price_score}")
print(f"Facility Consistency     : {facility_score}")
print("----------------------------------------")
print(f"FINAL SANITY SCORE       : {final_score} / 100")
print(f"STATUS                   : {label}")
print("========================================")
