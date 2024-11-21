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
```
This will train the model, display performance metrics, and save a decision tree visualization as 'decision_tree_visualization.png'.

## Feature Descriptions and Importance

The dataset contains the following features, listed in order of importance for heart disease prediction:

1. **Exercise Induced Angina** (42.57% importance)
   - Technical key: `exang`
   - Binary value indicating if exercise causes chest pain
   - Values: 0 = No, 1 = Yes

2. **ST Depression** (19.50% importance)
   - Technical key: `oldpeak`
   - ST depression induced by exercise relative to rest
   - Numerical value measuring ECG changes during stress test

3. **Number of Major Vessels** (12.83% importance)
   - Technical key: `ca`
   - Number of major blood vessels colored by fluoroscopy
   - Values: 0-3

4. **Cholesterol Level** (9.82% importance)
   - Technical key: `chol`
   - Serum cholesterol in mg/dl
   - Numerical value

5. **Resting Blood Pressure** (8.86% importance)
   - Technical key: `trestbps`
   - Resting blood pressure in mm Hg
   - Numerical value

Additional features include:

6. **Chest Pain Type** 
   - Technical key: `cp`
   - Values:
     - 0 = Typical Angina
     - 1 = Atypical Angina
     - 2 = Non-Anginal Pain
     - 3 = Asymptomatic

7. **Gender**
   - Technical key: `sex`
   - Values: 0 = Female, 1 = Male

8. **Resting ECG Results**
   - Technical key: `restecg`
   - Values:
     - 0 = Normal
     - 1 = ST-T Wave Abnormality
     - 2 = Left Ventricular Hypertrophy

9. **ST Slope**
   - Technical key: `slope`
   - The slope of the peak exercise ST segment
   - Values:
     - 0 = Upsloping
     - 1 = Flat
     - 2 = Downsloping

10. **Thalassemia Type**
    - Technical key: `thal`
    - Values:
      - 0 = Normal
      - 1 = Fixed Defect
      - 2 = Reversible Defect
      - 3 = Unknown

11. **Age**
    - Technical key: `age`
    - Age in years
    - Numerical value

12. **Maximum Heart Rate**
    - Technical key: `thalach`
    - Maximum heart rate achieved during exercise test
    - Numerical value

13. **Fasting Blood Sugar**
    - Technical key: `fbs`
    - Values: 0 = ≤120 mg/dl, 1 = >120 mg/dl

The importance values are derived from the decision tree model's feature importance scores and indicate how influential each feature is in predicting heart disease. The top 5 features account for over 93% of the model's decision-making process.