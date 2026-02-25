import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Wczytanie danych
df = pd.read_csv('dane_owocow.csv')

# 2. Podział na cechy (X) i wynik (y)
X = df[['waga', 'tekstura', 'kolor', 'ksztalt']]  # To model dostaje do analizy
y = df['typ']                # To ma odgadnąć

# 3. Wybór algorytmu i trening
model = DecisionTreeClassifier(max_depth=5)
model.fit(X.values, y)

# 4. Eksport modelu do pliku .pkl (KLUCZOWE)
joblib.dump(model, 'model_owocow.pkl')
print("Model wytrenowany i zapisany jako model_owocow.pkl!")