import numpy as np

# Step 1: Define the data
data = np.array([
    [8, 11],
    [10, 5],
    [13, 7],
    [7, 14],
    [5, 6],
    [7, 6]
])

# Step 2: Standardize the data
# Subtract the mean of each feature (column)
mean_centered_data = data - np.mean(data, axis=0)

# Step 3: Calculate the covariance matrix
cov_matrix = np.cov(mean_centered_data, rowvar=False)  # rowvar=False to treat rows as observations

# Step 4: Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Step 5: Sort the eigenvalues and eigenvectors by decreasing order of eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Step 6: Select the top k eigenvectors (k=1 in this case to reduce dimension to 1)
k = 1
projection_matrix = sorted_eigenvectors[:, :k]

# Step 7: Project the original standardized data onto the space defined by the top k eigenvectors
projected_data = mean_centered_data.dot(projection_matrix)

# Output
print("Projected Data onto the first principal component:\n", projected_data)
