import numpy as np
import pandas as pd
from itertools import chain, combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import io
import os
import json

# Functions for clustering and calculating quality metrics
def compute_cluster_radius(cluster_center, cluster_points):
    if cluster_points.shape[0] == 0:
        return 0
    distances = euclidean_distances(cluster_points, cluster_center.reshape(1, -1))
    return np.max(distances)

def compute_cluster_diameter(cluster_points):
    if cluster_points.shape[0] == 0:
        return 0
    distances = euclidean_distances(cluster_points)
    return np.max(distances)

def compute_cluster_quality(cluster_centers, cluster_points):
    cluster_radii = [compute_cluster_radius(center, points) for center, points in zip(cluster_centers, cluster_points)]
    cluster_diameters = [compute_cluster_diameter(points) for points in cluster_points]
    avg_diameter = np.mean(cluster_diameters)
    avg_radius = np.mean(cluster_radii)
    return avg_diameter, avg_radius

def powerset(s):
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))

def compute_knn(X, k, subspace_indices):
    """Compute k-nearest neighbors for each point in the specified subspace."""
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X[:, subspace_indices])
    distances, indices = nbrs.kneighbors(X[:, subspace_indices])
    return indices

def calculate_subspace_quality(X, D, k):
    P = powerset(D)
    subspace_quality = {subspace: 0 for subspace in P}

    for subspace in P:
        subspace_indices = list(subspace)

        # Compute cluster quality in the current subspace
        kmeans = KMeans(n_clusters=5, random_state=2)
        kmeans.fit(X[:, subspace_indices])
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        cluster_points = [X[:, subspace_indices][cluster_labels == i] for i in range(len(cluster_centers))]
        cluster_quality = [compute_cluster_quality([center], [points]) for center, points in zip(cluster_centers, cluster_points)]
        
        # Compute average quality of clusters weighted by the number of points
        total_points = sum(len(points) for points in cluster_points)
        avg_quality = sum((len(points) / total_points) * quality[0] for quality, points in zip(cluster_quality, cluster_points))
        if avg_quality == 0:
            subspace_quality[subspace] = 0
        else:
            subspace_quality[subspace] = 1 / avg_quality  # Inverse of average diameter for quality

    return subspace_quality

def binary_vector_to_int(binary_vector):
    """Convert a binary vector to an integer."""
    return int("".join(str(int(x)) for x in binary_vector), 2)

def normalize_value(value, max_value):
    """Normalize a value between 0 and 1."""
    return value / max_value

# Implementing kNN ordering
def knn_ordering(knn_indices):
    visited = set()
    order = []

    def dfs(node):
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbor in knn_indices[node]:
                dfs(neighbor)

    for start_node in range(knn_indices.shape[0]):
        if start_node not in visited:
            dfs(start_node)

    return order

# Function to compute the Heidi matrix and store it in a JSON file with kNN ordering
def heidi_matrix_top_subspaces(X, D, k, top_subspaces, output_json_filepath="../image_matrix.json"):
    n = X.shape[0]
    subspace_map = {subspace: idx for idx, subspace in enumerate(top_subspaces)}
    max_value = (1 << len(top_subspaces)) - 1  # Maximum possible value for the binary vector

    heidi_data = []

    for subspace in top_subspaces:
        subspace_indices = list(subspace)
        knn_indices = compute_knn(X, k, subspace_indices)

        # Apply kNN ordering within the subspace
        ordered_indices = knn_ordering(knn_indices)

        for i in ordered_indices:
            binary_vector = np.zeros(len(top_subspaces), dtype=int)
            for j in knn_indices[i]:
                binary_vector[subspace_map[subspace]] = 1

                integer_value = binary_vector_to_int(binary_vector)
                normalized_value = normalize_value(integer_value, max_value)

                # Store the data points and their normalized value
                heidi_data.append({
                    "p": int(i),
                    "q": int(j),
                    "value": normalized_value
                })

    # Save the Heidi data to a JSON file
    with open(output_json_filepath, 'w') as json_file:
        json.dump(heidi_data, json_file, indent=2)
    
    print(f"Heidi matrix data saved as {output_json_filepath}")
    return heidi_data

# Functions for clustering and ordering
def sort_subspaces_by_cluster_quality(subspace_quality):
    return sorted(subspace_quality.items(), key=lambda x: x[1], reverse=True)

# Example usage
def main(filepath):
    # Load and preprocess your data
    _, file_extension = os.path.splitext(filepath)
        
    if os.path.basename(filepath) == 'Iris.csv':
        data = pd.read_csv(filepath).iloc[:, 1:-1]
    elif file_extension.lower() == '.csv':
        data = pd.read_csv(filepath).iloc[:, :-1]
    elif file_extension.lower() in ('.xls', '.xlsx'):
        data = pd.read_excel(filepath).iloc[:, :-1]
    else:
        raise ValueError("Unsupported file format")

    label_encoders = {}
    for column in data.select_dtypes(include=['object']):
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=5, random_state=2)
    kmeans.fit(scaled_data)
    cluster_labels = kmeans.labels_

    D = range(scaled_data.shape[1])
    k = 6

    subspace_quality = calculate_subspace_quality(scaled_data, D, k)

    # Get the top subspaces by quality
    sorted_subspaces = sort_subspaces_by_cluster_quality(subspace_quality)
    top_subspaces = [subspace for subspace, _ in sorted_subspaces[:10]]  # Adjust the number of top subspaces as needed
    print(top_subspaces)

    # Compute the Heidi matrix with kNN ordering and save the data as a JSON file
    heidi_data = heidi_matrix_top_subspaces(scaled_data, D, k, top_subspaces)

if __name__ == "__main__":
    main()
