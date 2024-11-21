# Aufgabe 1
Basierend auf den Datensätzen `heart_Disease_training.csv` und `heart_Disease_validation.csv`:
- Implementiert einen Entscheidungsbaum mit der Python Package scikit-learn mit dem Trainingsdatensatz
- Lasst euch den Entscheidungsbaum graphisch darstellen
- Benutzt das Model um Vorhersagen mit dem `heart_Disease_validation.csv` Datensatz zu machen und vergleicht die Ergebnisse. Wie gut schneidet euer Model ab? Gibt es Möglichkeiten das Model zu verbessern?
- Präsentiert eure Ergebnisse
- Infos zum Datensatz: Heart_Diseases
# Aufgabe 2 (wenn ihr den Dingen auf den Grund gehen wollt):
Basierend auf den Datensätzen `heart_Disease_training.csv` und `heart_Disease_validation.csv`:
- Implementiert den kompletten Entscheidungsbaum Algorithmus
- Nutzt euer Model zusammen mit dem Validierungsdatensatz um Vorhersagen zu treffen und vergleicht die Vorhersagen mit dem Datensatz
- Präsentiert eure Ergebnisse
- Infos zum Datensatz: Heart_Diseases

# Getting Started
To run this project, follow these steps:

1. Create a Python virtual environment:
```bash
python -m venv venv
```
2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```
3. Install required dependencies:
```bash
pip install -r requirements.txt
```
4. Run the decision tree classifier:
```bash
python heart_disease_classifier.py
# same for custom_decision_tree.py
```
This will:
1. Train the model and display performance metrics
2. Generate a detailed feature analysis log in 'feature_analysis.txt'
3. Save visualizations as 'decision_tree_visualization.png' and 'feature_importance.png'

## Feature Descriptions and Importance

The dataset contains the following features, listed in order of importance for heart disease prediction:

1. **Exercise Induced Angina** (42.57% importance)
   - Technical key: `exang`
   - Binary value indicating if exercise causes chest pain (angina pectoris)
   - Values: 0 = No, 1 = Yes

2. **ST Depression** (19.50% importance)
   - Technical key: `oldpeak`
   - ST segment depression induced by exercise relative to rest
   - ST refers to the segment in an electrocardiogram (ECG) between the S and T waves
   - Numerical value measuring ECG changes during stress test
   - Important indicator of reduced blood flow to the heart (ischemia)

3. **Number of Major Vessels** (12.83% importance)
   - Technical key: `ca` (stands for Coronary Arteries)
   - Number of major blood vessels colored by fluoroscopy
   - Values: 0-3 (number of vessels with significant narrowing)

4. **Cholesterol Level** (9.82% importance)
   - Technical key: `chol`
   - Serum cholesterol measured in mg/dl (milligrams per deciliter)
   - Numerical value
   - High levels may indicate increased risk of heart disease

5. **Resting Blood Pressure** (8.86% importance)
   - Technical key: `trestbps` (Target RESTing Blood PresSure)
   - Resting blood pressure in mm Hg (millimeters of mercury)
   - Numerical value
   - Measured when patient is at rest

Additional features include:

6. **Chest Pain Type** 
   - Technical key: `cp`
   - Values:
     - 0 = Typical Angina (classic chest pain indicating heart problems)
     - 1 = Atypical Angina (chest pain not typical of heart problems)
     - 2 = Non-Anginal Pain (chest pain not related to heart)
     - 3 = Asymptomatic (no chest pain)

7. **Gender**
   - Technical key: `sex`
   - Values: 0 = Female, 1 = Male

8. **Resting ECG Results**
   - Technical key: `restecg` (RESTing ElectroCardioGram)
   - Values:
     - 0 = Normal
     - 1 = ST-T Wave Abnormality (changes in ST segment and T waves)
     - 2 = LVH (Left Ventricular Hypertrophy: thickening of heart's left pumping chamber)

9. **ST Slope**
   - Technical key: `slope`
   - The slope of the peak exercise ST segment in ECG
   - Describes how the ST segment changes during exercise
   - Values:
     - 0 = Upsloping (positive angle)
     - 1 = Flat (horizontal)
     - 2 = Downsloping (negative angle)
   - Important indicator of heart stress response

10. **Thalassemia Type**
    - Technical key: `thal`
    - A blood disorder that affects hemoglobin production
    - Values:
      - 0 = Normal
      - 1 = Fixed Defect (permanent blood flow abnormality)
      - 2 = Reversible Defect (temporary blood flow abnormality)
      - 3 = Unknown

11. **Age**
    - Technical key: `age`
    - Age in years
    - Numerical value

12. **Maximum Heart Rate**
    - Technical key: `thalach` (THALium stress test maximum Heart rate)
    - Maximum heart rate achieved during exercise test
    - Numerical value
    - Important indicator of cardiovascular fitness

13. **Fasting Blood Sugar**
    - Technical key: `fbs`
    - Blood sugar levels after fasting
    - Values: 0 = ≤120 mg/dl, 1 = >120 mg/dl
    - High levels may indicate diabetes, a risk factor for heart disease

### Medical Terms Explained

- **Angina**: Chest pain caused by reduced blood flow to the heart
- **ECG/EKG**: Electrocardiogram, a test that measures the electrical activity of the heart
- **ST Segment**: A portion of the ECG wave between the S and T waves, representing the period when the ventricles are depolarized
- **Fluoroscopy**: An X-ray based imaging technique that shows real-time images of the internal structures
- **LVH**: Left Ventricular Hypertrophy, a condition where the left ventricle becomes thickened
- **Thalassemia**: An inherited blood disorder that affects the body's ability to produce hemoglobin
- **mm Hg**: Millimeters of mercury, the standard unit for measuring blood pressure
- **mg/dl**: Milligrams per deciliter, a unit of measurement for blood sugar and cholesterol levels

The importance values are derived from the decision tree model's feature importance scores and indicate how influential each feature is in predicting heart disease. The top 5 features account for over 93% of the model's decision-making process.

## Custom Decision Tree Implementation (Exercise 2)

A custom implementation of the decision tree algorithm is provided in `custom_decision_tree.py`. This implementation:

1. Builds a decision tree from scratch using the CART (Classification and Regression Trees) algorithm
2. Uses Gini impurity as the splitting criterion
3. Implements feature importance calculation
4. Provides visualization capabilities

### Key Components

- `Node` class: Represents a node in the decision tree with:
  - Feature index for splitting
  - Threshold value
  - Left and right child nodes
  - Predicted class value (for leaf nodes)

- `CustomDecisionTree` class: Main implementation with methods for:
  - Tree construction (`fit`)
  - Making predictions (`predict`)
  - Finding optimal splits (`_best_split`)
  - Calculating information gain (`_information_gain`)
  - Computing Gini impurity (`_gini`)
  - Visualizing feature importance (`plot_feature_importance`)

### Running the Custom Implementation

```bash
python custom_decision_tree.py
```

This will:
1. Train a decision tree using the custom implementation
2. Make predictions on the validation dataset
3. Display the model's accuracy
4. Generate a feature importance plot as 'custom_feature_importance.png'

### Comparison with Scikit-learn Implementation

You can compare the results of both implementations:
- Accuracy scores
- Feature importance rankings
- Model complexity and training time
- Prediction patterns

The custom implementation provides insights into the core mechanics of decision trees, while the scikit-learn version offers a more optimized and production-ready solution.

### Implementation Comparison Results

Both implementations were run on the same dataset with similar parameters (max_depth=6):

#### Model Performance

| Metric                          | Scikit-learn Implementation | Custom Implementation |
|--------------------------------|----------------------------|---------------------|
| Validation Set Accuracy         | 42%                       | 41%                |
| Cross-Validation Score         | 82%                       | Not Implemented    |
| Training Time                   | Faster                    | Slower             |

#### Top 5 Features by Importance

Scikit-learn Implementation:
1. Exercise Induced Angina (42.57%)
2. ST Depression (19.50%)
3. Number of Major Vessels (12.83%)
4. Cholesterol Level (9.82%)
5. Resting Blood Pressure (8.86%)

Custom Implementation (based on split frequency):
1. Exercise Induced Angina
2. ST Depression
3. Number of Major Vessels
4. Maximum Heart Rate
5. Age

#### Key Differences

1. **Performance**:
   - Both implementations achieve similar accuracy on the validation set
   - The scikit-learn version includes additional optimizations and cross-validation capabilities
   - The custom implementation is more straightforward but less optimized

2. **Feature Importance**:
   - Scikit-learn uses a more sophisticated importance calculation based on impurity reduction
   - Custom implementation uses split frequency as a simpler proxy for importance
   - Both identify similar top features, particularly the top 3

3. **Implementation Details**:
   - Scikit-learn includes additional optimizations and features:
     - Efficient memory usage
     - Multi-threading support
     - Advanced pruning techniques
   - Custom implementation focuses on core algorithm components:
     - Basic CART algorithm
     - Gini impurity
     - Binary splitting

4. **Visualization**:
   - Both implementations provide feature importance visualization
   - Scikit-learn offers additional visualization options (e.g., tree structure)
   - Custom implementation focuses on basic feature importance plotting

The similar performance between implementations validates the correctness of our custom version, while the differences highlight the additional optimizations in the scikit-learn library that make it more suitable for production use.