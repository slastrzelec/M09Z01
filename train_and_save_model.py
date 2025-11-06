import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Wczytanie danych
print("Wczytywanie danych...")
df = pd.read_csv(r"C:\Users\slast\PYTHON\AI\M09\Zad_1\df_ml_clean.csv", sep=";")
print(f"✅ Wczytano {len(df)} rekordów")

# Kodowanie zmiennej kategorycznej
print("\nKodowanie zmiennych...")
le = LabelEncoder()
df['plec_encoded'] = le.fit_transform(df['Płeć'])

# Przygotowanie danych
X = df[['plec_encoded', '5 km Tempo']]
y = df['Czas [s]']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Trenowanie modelu
print("\nTrenowanie modelu...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model wytrenowany")

# Predykcja i ocena
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*50)
print("WYNIKI MODELU")
print("="*50)
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f} sekund ({rmse/60:.2f} minut)")
print(f"MAE: {mae:.2f} sekund ({mae/60:.2f} minut)")
print("="*50)

# Zapisanie modelu i encodera
print("\nZapisywanie modelu...")
model_data = {
    'model': model,
    'label_encoder': le,
    'r2_score': r2,
    'rmse': rmse,
    'mae': mae,
    'train_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('half_marathon_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✅ Model zapisany jako 'half_marathon_model.pkl'")
print("\nMożesz teraz uruchomić aplikację Streamlit!")

# Przykładowa predykcja
print("\n" + "="*50)
print("PRZYKŁADOWE PREDYKCJE")
print("="*50)

# Test dla mężczyzny
plec_encoded = le.transform(['M'])[0]
dane_test_m = pd.DataFrame({
    'plec_encoded': [plec_encoded],
    '5 km Tempo': [5.5]
})
pred_m = model.predict(dane_test_m)[0]
print(f"Mężczyzna, tempo 5.5 min/km: {pred_m:.0f} sekund ({pred_m/60:.1f} minut)")

# Test dla kobiety
plec_encoded = le.transform(['K'])[0]
dane_test_k = pd.DataFrame({
    'plec_encoded': [plec_encoded],
    '5 km Tempo': [6.5]
})
pred_k = model.predict(dane_test_k)[0]
print(f"Kobieta, tempo 6.5 min/km: {pred_k:.0f} sekund ({pred_k/60:.1f} minut)")
print("="*50)