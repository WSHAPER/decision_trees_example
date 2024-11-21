import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from feature_mapping import get_feature_display_name, get_all_feature_names

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
    'feature': [get_feature_display_name(col) for col in X_train.columns],
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Write detailed feature analysis to a log file
with open('outputs/feature_analysis.txt', 'w') as f:
    f.write("Heart Disease Classifier - Feature Analysis\n")
    f.write("=" * 50 + "\n\n")
    
    # Model parameters
    f.write("Model Configuration\n")
    f.write("-" * 20 + "\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best Cross-Validation Score (Balanced Accuracy): {grid_search.best_score_:.2f}\n")
    f.write(f"Validation Set Accuracy: {accuracy:.2f}\n\n")
    
    # Feature importance analysis
    f.write("Feature Importance Analysis\n")
    f.write("-" * 25 + "\n")
    f.write("Features ranked by importance in model predictions:\n\n")
    
    total_importance = 0
    for idx, (feature, importance) in enumerate(zip(feature_importance['feature'], feature_importance['importance']), 1):
        total_importance += importance
        f.write(f"{idx}. {feature:<25} {importance:.4f} ({importance*100:.2f}%)\n")
        f.write(f"   Cumulative importance: {total_importance*100:.2f}%\n\n")
    
    # Dataset statistics
    f.write("\nDataset Statistics\n")
    f.write("-" * 17 + "\n")
    f.write(f"Training set shape: {train_data.shape}\n")
    f.write("Training set class distribution:\n")
    train_dist = train_data['target'].value_counts(normalize=True)
    f.write(f"No Disease (0): {train_dist[0]:.2%}\n")
    f.write(f"Disease (1): {train_dist[1]:.2%}\n\n")
    
    f.write(f"Validation set shape: {validation_data.shape}\n")
    f.write("Validation set class distribution:\n")
    val_dist = validation_data['target'].value_counts(normalize=True)
    for label in sorted(val_dist.index):
        f.write(f"Class {label}: {val_dist[label]:.2%}\n")
    
    # Performance metrics
    f.write("\nModel Performance Metrics\n")
    f.write("-" * 23 + "\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_val, y_pred))
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\nNote: Feature importance values indicate the relative contribution\n")
    f.write("of each feature to the model's predictions. Higher values indicate\n")
    f.write("stronger influence on the decision-making process.\n")

print("\nFeature analysis has been saved to 'outputs/feature_analysis.txt'")

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png')

# Visualize the decision tree (limited depth for clarity)
plt.figure(figsize=(20,10))
plot_tree(best_model, 
          feature_names=[get_feature_display_name(col) for col in X_train.columns],
          class_names=['No Disease', 'Disease'],
          filled=True, rounded=True, max_depth=3)
plt.tight_layout()
plt.savefig('outputs/decision_tree_visualization.png')
print("\nVisualizations have been saved as 'outputs/decision_tree_visualization.png' and 'outputs/feature_importance.png'")
