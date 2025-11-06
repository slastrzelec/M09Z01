# model_tre_3x.py
import pandas as pd
from pycaret.regression import setup, create_model, finalize_model, predict_model, plot_model

# 🔹 Wczytaj dane
df = pd.read_csv(r"C:\Users\slast\PYTHON\AI\M09\Zad_1\df_ml_clean.csv", sep=';')

# 🔹 Sprawdzenie danych
print("Typy danych w kolumnach:")
print(df.dtypes)
print("\nPierwsze 5 wierszy:")
print(df.head())

# 🔹 Wybór kolumny docelowej
target_col = 'Czas [s]'

# 🔹 Setup środowiska PyCaret 3.x
reg_setup = setup(
    data=df,
    target=target_col,
    categorical_features=['Płeć'],  # wskazujemy kolumny kategoryczne
    numeric_features=['5 km Tempo'], # opcjonalnie wskazujemy kolumny numeryczne
    session_id=123,
    verbose=False,
    interactive=False
)

# 🔹 Tworzymy wybrany model regresji (np. Random Forest)
model = create_model('rf')

# 🔹 Finalizujemy model
final_model = finalize_model(model)

# 🔹 Predykcje na tym samym zbiorze (dla demonstracji)
predictions = predict_model(final_model, data=df)

# 🔹 Wyświetlenie wyników
print("\nPredykcje dla pierwszych 5 wierszy:")
print(predictions.head())

# 🔹 Wykres ważności cech
plot_model(final_model, plot='feature')
