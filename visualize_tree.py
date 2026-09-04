import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("AIMLT5\\data\\heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

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
# 3. TRAIN DECISION TREE
# ==========================================

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 4. CREATE OUTPUT DIRECTORY
# ==========================================

os.makedirs("AIMLT5\\outputs", exist_ok=True)

# =============================================
# 5. VISUALIZE TREE
# ==========================================

plt.figure(figsize=(22, 12))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Heart Disease Classification")

plt.savefig(
    "AIMLT5\\outputs\\decision_tree.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Decision tree saved to:")
print("AIMLT5\\outputs\\decision_tree.png")