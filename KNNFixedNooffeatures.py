import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset
df = pd.read_csv("C:/Users/User/Downloads/orange_dataset.csv")  # Use full path if running locally

# Manhattan distance function for two fixed features: Size and Texture
def manhattan_distance(x1, x2):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

# Fixed-feature KNN function
def knn_classify_fixed(new_item, k, dataset):
    distances = []
    for i, row in dataset.iterrows():
        features = [row['Size'], row['Texture']]  # fixed feature access
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

# Test the KNN function with a new orange
new_orange = [2, 0]
k = 3
predicted_class, distance_list, top_categories = knn_classify_fixed(new_orange, k, df)

# Visualize top 10 distances
plt.figure(figsize=(10, 5))
top_n = 10
top_indices = [i for i, _ in distance_list[:top_n]]
x_labels = [f"Index {i}" for i in top_indices]  # fixed: using DataFrame index
y_values = [dist for _, dist in distance_list[:top_n]]
top_k_categories = [df.iloc[i]['Category'] for i in top_indices]

# Color bars: green for k-nearest, orange for rest
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
