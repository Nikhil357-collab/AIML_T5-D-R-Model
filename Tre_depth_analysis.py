import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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
# 3. TEST DIFFERENT TREE DEPTHS
# ==========================================

depths = range(1, 16)

train_scores = []
test_scores = []

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_pred
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    train_scores.append(train_accuracy)
    test_scores.append(test_accuracy)

# ==========================================
# 4. PRINT RESULTS
# ==========================================

print("=" * 60)
print("TREE DEPTH ANALYSIS")
print("=" * 60)

for depth, train_acc, test_acc in zip(
    depths,
    train_scores,
    test_scores
):
    print(
        f"Depth {depth:2d} | "
        f"Train Accuracy: {train_acc:.4f} | "
        f"Test Accuracy: {test_acc:.4f}"
    )

# ==========================================
# 5. FIND BEST DEPTH
# ==========================================

best_index = test_scores.index(max(test_scores))
best_depth = list(depths)[best_index]

print("\nBest Tree Depth:", best_depth)
print("Best Test Accuracy:", test_scores[best_index])

# ==========================================
# 6. VISUALIZE OVERFITTING
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    depths,
    train_scores,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depths,
    test_scores,
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Depth vs Accuracy")

plt.legend()
plt.grid(True)

plt.show()

plt.savefig("AIMLT5\\outputs\\tree_depth_analysis.png", dpi=300, bbox_inches="tight")
print("Tree depth analysis saved to:")
print("AIMLT5\\outputs\\tree_depth_analysis.png")