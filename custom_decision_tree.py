"""
Custom implementation of a Decision Tree Classifier from scratch.
This implementation follows the basic CART (Classification and Regression Trees) algorithm.
"""

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from feature_mapping import get_feature_display_name
import os

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Predicted class (for leaf nodes)

class CustomDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_importance_ = None
        self.n_features_ = None

    def fit(self, X, y):
        """Train the decision tree."""
        self.n_features_ = X.shape[1]
        self.feature_importance_ = np.zeros(self.n_features_)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:  # No valid split found
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Update feature importance
        self.feature_importance_[best_feature] += 1

        # Create child splits
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        """Find the best split using Gini impurity."""
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features_):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        """Calculate information gain using Gini impurity."""
        parent_gini = self._gini(y)

        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = sum(left_idxs), sum(right_idxs)
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_gini - child_gini

    def _gini(self, y):
        """Calculate Gini impurity."""
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum([p ** 2 for p in proportions])

    def _most_common_label(self, y):
        """Return the most common class label."""
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Predict class for X."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction."""
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def plot_feature_importance(self, feature_names=None):
        """Plot feature importance."""
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(self.n_features_)]

        # Create output directory if it doesn't exist
        output_dir = 'outputs/custom_decision_tree'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        importance = self.feature_importance_ / np.sum(self.feature_importance_)
        sorted_idx = np.argsort(importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        plt.figure(figsize=(12, 6))
        plt.barh(pos, importance[sorted_idx])
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importance (Custom Implementation)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'custom_feature_importance.png'))
        plt.close()

if __name__ == '__main__':
    # Load the datasets
    train_data = pd.read_csv('sample_data/heart_Disease_training.csv')
    validation_data = pd.read_csv('sample_data/heart_Disease_validation.csv')

    # Prepare the data
    X_train = train_data.drop('target', axis=1).values
    y_train = train_data['target'].values
    X_val = validation_data.drop('target', axis=1).values
    y_val = validation_data['target'].values

    # Train the custom decision tree
    custom_dt = CustomDecisionTree(max_depth=6, min_samples_split=2)
    custom_dt.fit(X_train, y_train)

    # Make predictions
    y_pred = custom_dt.predict(X_val)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_val)
    print(f"\nCustom Decision Tree Results:")
    print(f"Accuracy on validation set: {accuracy:.2f}")

    # Plot feature importance with human-readable names
    feature_names = [get_feature_display_name(col) for col in train_data.drop('target', axis=1).columns]
    custom_dt.plot_feature_importance(feature_names)
    print("\nFeature importance plot has been saved as 'outputs/custom_decision_tree/custom_feature_importance.png'")
