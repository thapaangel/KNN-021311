import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:/Users/User/Downloads/orange_dataset.csv")  # Adjust if needed

# Manhattan distance function
def manhattan_distance(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

# Generalized KNN function
def knn_classify(new_item, k, dataset, feature_columns):
    distances = []
    for i, row in dataset.iterrows():
        features = [row[col] for col in feature_columns]
        dist = manhattan_distance(new_item, features)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    top_k = distances[:k]
    categories = [dataset.iloc[i]['Category'] for i, _ in top_k]
    freq = Counter(categories)
    prediction = freq.most_common(1)[0][0]

    print(f"\nNew Item: {new_item}")
    print(f"Predicted Category: {prediction}")
    return prediction, distances, categories

# Test with new input
feature_columns = ['Size', 'Texture']
new_orange = [2, 0]
k = 5

predicted_class, distance_list, top_categories = knn_classify(new_orange, k, df, feature_columns)

# Visualize distances (Top 10)
plt.figure(figsize=(10, 5))
top_n = 10
top_indices = [i for i, _ in distance_list[:top_n]]

# ❌ Original x_labels line used non-existent 'ID' column
# ✅ Corrected: use DataFrame index
x_labels = [f"Index {i}" for i in top_indices]
y_values = [dist for _, dist in distance_list[:top_n]]
top_k_categories = [df.iloc[i]['Category'] for i in top_indices]

# Color bars: green for top-k, orange for others
colors = ['green' if i < k else 'orange' for i in range(top_n)]
bars = plt.bar(x_labels, y_values, color=colors)

# Add category labels above bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
             top_k_categories[i], ha='center', fontsize=9)

plt.title("Top 10 Manhattan Distances to New Orange")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.xticks(rotation=45)
plt.ylim(0, max(y_values) * 1.3)
plt.tight_layout()
plt.show()
