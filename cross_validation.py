import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("AIMLT5\\data\\heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ==========================================
# 2. DEFINE CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# ==========================================
# 3. DECISION TREE
# ==========================================

dt_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

dt_scores = cross_val_score(
    dt_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# ==========================================
# 4. RANDOM FOREST
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

# ==========================================
# 5. RESULTS
# ==========================================

print("=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

print("\nDecision Tree CV Scores:")

for i, score in enumerate(dt_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print(
    "Mean Accuracy:",
    round(dt_scores.mean(), 4)
)

print(
    "Standard Deviation:",
    round(dt_scores.std(), 4)
)

print("\nRandom Forest CV Scores:")

for i, score in enumerate(rf_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print(
    "Mean Accuracy:",
    round(rf_scores.mean(), 4)
)

print(
    "Standard Deviation:",
    round(rf_scores.std(), 4)
)