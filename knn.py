import numpy as np
from collections import Counter


def euclidean_distance(instance1, instance2):
    """
    Calculate the Euclidean distance between two data points.
    Parameters:
    - instance1: np.array, first data point
    - instance2: npome array, second data .point

    Returns:
    - float, the Euclidean distance
    """
    return np.sqrt(np.sum((instance1 - instance2) ** 2))


def get_nearest_neighbors(training_data, test_instance, k):
    """
    Identify the k nearest neighbors to the test_instance using the training data.
    Parameters:
    - training_data: list of tuples, where each tuple is (features_array, label)
    - test_instance: np.array, the data point for which to find neighbors
    - k: int, the number of neighbors to return

    Returns:
    - list of tuples, the k nearest neighbors (features, label)
    """
    distances = []
    for instance in training_data:
        dist = euclidean_distance(test_instance, instance[0])
        distances.append((instance, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

def majority_voting(neighbors):
    """
    Determine the most common class among the neighbors by majority vote.
    Parameters:
    - neighbors: list of tuples, each tuple contains (features_array, label)

    Returns:
    - string, the predicted class label
    """
    classes = [neighbor[1] for neighbor in neighbors]  # Extract the labels
    count = Counter(classes)
    return count.most_common(1)[0][0]


def knn_classify(test_instance, training_data, k):
    """
    Classify a test instance using the k-nearest neighbors algorithm.
    Parameters:
    - test_instance: np.array, the instance to classify
    - training_data: list of tuples, where each tuple is (features_array, label)
    - k: int, the number of neighbors to consider

    Returns:
    - string, the predicted class label
    """
    neighbors = get_nearest_neighbors(training_data, test_instance, k)
    prediction = majority_voting(neighbors)
    return prediction


# Example of using the KNN implementation
if __name__ == "__main__":
    # Creating a dataset (features, class)
    # dataset = [
    #     (np.array([5.3, 3.7]), 'Setosa'),
    #     (np.array([5.1, 3.8]), 'Setosa'),
    #     (np.array([7.2, 3.0]), 'Virginica'),
    #     (np.array([5.4, 3.4]), 'Setosa'),
    #     (np.array([5.1, 3.3]), 'Setosa'),
    #     (np.array([5.4, 3.9]), 'Setosa'),
    #     (np.array([7.4, 2.8]), 'Virginica'),
    #     (np.array([6.1, 2.8]), 'Virginica'),
    #     (np.array([7.3, 2.9]), 'Virginica'),
    #     (np.array([6.0, 2.7]), 'Versicolor'),
    #     (np.array([5.8, 2.8]), 'Versicolor'),
    #     (np.array([6.3, 2.3]), 'Versicolor'),
    #     (np.array([5.1, 2.5]), 'Versicolor'),
    #     (np.array([6.3, 2.5]), 'Versicolor'),
    #     (np.array([5.5, 2.4]), 'Versicolor')
    # ]
    dataset = [
        (np.array([167, 51]), 'Underweight'),
        (np.array([182, 62]), 'Normal'),
        (np.array([176, 69]), 'Normal'),
        (np.array([173, 64]), 'Normal'),
        (np.array([172, 65]), 'Normal'),
        (np.array([174, 56]), 'Underweight'),
        (np.array([169, 58]), 'Normal'),
        (np.array([173, 57]), 'Normal'),
        (np.array([170, 55]), 'Normal')
    ]

    # Test instance
    # test_instance = np.array([5.2,3.1])

    test_instance = np.array([140, 47])
    # Number of neighbors
    k = 2

    # Predict the class for the test instance
    prediction = knn_classify(test_instance, dataset, k)
    print("Predicted Class:", prediction)
