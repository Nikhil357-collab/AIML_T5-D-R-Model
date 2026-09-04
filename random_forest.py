import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("AIMLT5\\data\\heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ==========================================
# 2. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 3. DECISION TREE
# ==========================================

dt_model = DecisionTreeClassifier(
    max_depth=9,
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

# ==========================================
# 4. RANDOM FOREST
# ==========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

# ==========================================
# 5. COMPARISON
# ==========================================

print("=" * 60)
print("DECISION TREE VS RANDOM FOREST")
print("=" * 60)

print(f"Decision Tree Accuracy : {dt_accuracy:.4f}")
print(f"Random Forest Accuracy : {rf_accuracy:.4f}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))