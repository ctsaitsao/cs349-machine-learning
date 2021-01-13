import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.
            assignments (numpy array of (n_samples,)): A numpy array containing the cluster labels
                for the generated data
            means (list of numpy arrays): Collection of means of dimension of n_features.

        """
        self.n_clusters = n_clusters
        self.assignments = None
        self.means = None

    def __update_assignments(self, features):
        for feature in features:
            distances = []
            for mean in self.means:
                distances.append(np.linalg.norm(feature - mean))
            label = distances.index(min(distances))
            self.assignments[np.where(features == feature)[0]] = label

    def __update_means(self, features):
        examples = np.concatenate((features, np.vstack(self.assignments)), axis=1)
        for i in range(self.n_clusters):
            cluster_examples = examples[examples[:, -1] == i]
            cluster_features = cluster_examples[:, :-1]
            self.means[i] = np.zeros(features.shape[1])
            for j in range(features.shape[1]):
                self.means[i][j] = np.sum(cluster_features[:, j]) / cluster_features.shape[0]

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        self.assignments = np.zeros(features.shape[0])
        # for i in range(20):
        #     print(np.random.choice(features.shape[0], replace=False))
        self.means = [features[np.random.choice(features.shape[0], replace=False), :] for i in range(self.n_clusters)]

        while True:
            prev_assignments = self.assignments.copy()
            self.__update_assignments(features)
            if (self.assignments == prev_assignments).all():
                break
            else:
                self.__update_means(features)

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        predictions = np.zeros(features.shape[0])

        for feature in features:
            distances = []
            for mean in self.means:
                distances.append(np.linalg.norm(feature - mean))
            label = distances.index(min(distances))
            predictions[np.where(features == feature)[0]] = label
        return predictions
