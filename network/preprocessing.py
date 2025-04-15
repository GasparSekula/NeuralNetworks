import numpy as np

def classes_to_one_hot(labels, num_classes=None):
    """
    Convert class labels to one-hot encoding.

    Args:
        labels (np.ndarray): Array of class labels.
        num_classes (int, optional): Total number of classes. If None, it will be inferred from the labels.

    Returns:
        np.ndarray: One-hot encoded array.
    """
    labels = labels.flatten()
    if num_classes is None:
        num_classes = np.max(labels) + 1
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

class StandardScaler:
    """
    StandardScaler for normalizing data.
    """

    def fit(self, X):
        """
        Fit the scaler to the data.

        Args:
            X (np.ndarray): Data to fit.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        """
        Transform the data using the fitted scaler.

        Args:
            X (np.ndarray): Data to transform.

        Returns:
            np.ndarray: Scaled data.
        """
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """
        Fit the scaler to the data and transform it.

        Args:
            X (np.ndarray): Data to fit and transform.

        Returns:
            np.ndarray: Scaled data.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """
        Inverse transform the scaled data.

        Args:
            X (np.ndarray): Scaled data to inverse transform.

        Returns:
            np.ndarray: Original data.
        """
        return X * self.std + self.mean
    
