import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
# 3. CREATE DECISION TREE
# ==========================================

model = DecisionTreeClassifier(
    random_state=42
)

# 4. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

# ==========================================
# 5. PREDICTION

y_pred = model.predict(X_test)

# 6. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("DECISION TREE CLASSIFIER")
print("=" * 60)

print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy :", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))