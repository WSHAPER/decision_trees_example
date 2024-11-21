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