import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Load the datasets
train_data = pd.read_csv('sample_data/heart_Disease_training.csv')
validation_data = pd.read_csv('sample_data/heart_Disease_validation.csv')

# Print data distribution
print("\nData Distribution:")
print("Training set shape:", train_data.shape)
print("Training set class distribution:\n", train_data['target'].value_counts(normalize=True))
print("\nValidation set shape:", validation_data.shape)
print("Validation set class distribution:\n", validation_data['target'].value_counts(normalize=True))

# Separate features and target
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_val = validation_data.drop('target', axis=1)
y_val = validation_data['target']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Calculate class weights
class_weights = dict(zip(
    np.unique(y_train),
    len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
))

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3],
    'criterion': ['gini', 'entropy']
}

# Create base classifier with class weights
dt_classifier = DecisionTreeClassifier(random_state=42, class_weight=class_weights)

# Perform grid search with stratified cross-validation
grid_search = GridSearchCV(
    estimator=dt_classifier,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='balanced_accuracy',
    n_jobs=-1
)

# Fit the grid search
grid_search.fit(X_train_scaled, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_val_scaled)

# Calculate metrics
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)

print(f"\nBest Model Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score (Balanced Accuracy): {grid_search.best_score_:.2f}")
print(f"Model Accuracy on Validation Set: {accuracy:.2f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Visualize the decision tree (limited depth for clarity)
plt.figure(figsize=(20,10))
plot_tree(best_model, feature_names=list(X_train.columns), class_names=['No Disease', 'Disease'], 
          filled=True, rounded=True, max_depth=3)
plt.savefig('decision_tree_visualization.png')
print("\nVisualizations have been saved as 'decision_tree_visualization.png' and 'feature_importance.png'")
