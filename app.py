import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

# ✅ Konfiguracja strony (musi być tylko raz i jako pierwsze polecenie Streamlit)
st.set_page_config(
    page_title="Predykcja czasu półmaratonu",
    page_icon="🏃",
    layout="wide"
)

# 🔹 Ustaw host i port zgodnie z wymaganiami DigitalOcean
port = int(os.environ.get("PORT", 8501))  # domyślnie 8501 lokalnie

# ===============================
# FUNKCJA WCZYTANIA MODELU
# ===============================
@st.cache_resource
def load_model():
    """Wczytuje wytrenowany model z pliku"""
    try:
        model_path = 'half_marathon_model.pkl'
        
        if not os.path.exists(model_path):
            return {
                'success': False,
                'error': f"Plik modelu '{model_path}' nie został znaleziony. Najpierw uruchom skrypt trenowania modelu."
            }
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return {
            'success': True,
            'model': model_data['model'],
            'le': model_data['label_encoder'],
            'r2': model_data['r2_score'],
            'rmse': model_data['rmse'],
            'mae': model_data['mae'],
            'train_samples': model_data['train_samples'],
            'test_samples': model_data['test_samples']
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Błąd wczytywania modelu: {str(e)}"
        }

# ===============================
# WCZYTANIE MODELU
# ===============================
model_data = load_model()

# ===============================
# GŁÓWNY NAGŁÓWEK
# ===============================
st.title("🏃 Predyktor czasu półmaratonu")
st.markdown("### Przewiduj swój czas na podstawie płci i tempa na 5km")
st.markdown("---")

# ===============================
# SPRAWDZENIE MODEL DATA
# ===============================
if not model_data['success']:
    st.error(f"❌ {model_data['error']}")
    st.info("💡 Uruchom najpierw skrypt 'train_model.py' aby wytrenować i zapisać model.")
    st.stop()

# ===============================
# SIDEBAR — INFORMACJE O MODELU
# ===============================
with st.sidebar:
    st.header("ℹ️ O modelu")
    st.success("✅ Model wczytany pomyślnie")
    st.markdown("---")
    st.subheader("📊 Jakość modelu")
    st.metric("R² Score", f"{model_data['r2']:.4f}", help="Współczynnik determinacji - miara dopasowania modelu")
    st.metric("RMSE", f"{model_data['rmse']:.0f} sek", help="Średni błąd predykcji")
    st.metric("MAE", f"{model_data['mae']:.0f} sek", help="Średni błąd bezwzględny")
    st.markdown("---")
    st.subheader("🎓 Dane treningowe")
    st.info(f"**Próbki treningowe:** {model_data['train_samples']}\n\n**Próbki testowe:** {model_data['test_samples']}")
    st.markdown("---")
    st.markdown("### 🏃‍♂️ Jak to działa?")
    st.markdown("""
    Model uczenia maszynowego przewiduje czas ukończenia półmaratonu na podstawie:
    - Twojej płci  
    - Twojego tempa na 5 km  
    """)

# ===============================
# GŁÓWNA CZĘŚĆ STRONY
# ===============================
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Wprowadź swoje dane")

    with st.form("prediction_form"):
        plec = st.selectbox(
            "Płeć",
            options=['K', 'M'],
            format_func=lambda x: "👩 Kobieta" if x == 'K' else "👨 Mężczyzna"
        )

        tempo_5km = st.number_input(
            "Tempo na 5 km (min/km)",
            min_value=3.0,
            max_value=15.0,
            value=6.0,
            step=0.1,
            help="Podaj swoje średnie tempo biegu na dystansie 5 km"
        )

        st.caption(f"💡 Dla tempa {tempo_5km} min/km, czas na 5km wynosi: {tempo_5km * 5:.1f} minut")

        submit_button = st.form_submit_button("🔮 Przewidź czas półmaratonu", use_container_width=True)

    if submit_button:
        model = model_data['model']
        le = model_data['le']
        plec_encoded = le.transform([plec])[0]
        dane_wejsciowe = pd.DataFrame({
            'plec_encoded': [plec_encoded],
            '5 km Tempo': [tempo_5km]
        })
        predykcja = model.predict(dane_wejsciowe)[0]
        st.session_state['last_prediction'] = {
            'plec': plec,
            'tempo': tempo_5km,
            'czas_sek': predykcja
        }

    with st.expander("ℹ️ Wskazówki dotyczące tempa"):
        st.markdown("""
        **Jak określić swoje tempo na 5km?**
        - Użyj swojego przeciętnego tempa z biegu 5km  
        - Dla przykładu:
            - 4–5 min/km: zaawansowani  
            - 6–7 min/km: amatorzy  
            - 8–9 min/km: początkujący  
        """)

with col2:
    st.header("🎯 Twój przewidywany czas")

    if 'last_prediction' in st.session_state:
        pred = st.session_state['last_prediction']
        st.success("✅ Predykcja gotowa!")

        czas_sek = pred['czas_sek']
        godziny = int(czas_sek // 3600)
        minuty = int((czas_sek % 3600) // 60)
        sekundy = int(czas_sek % 60)

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 40px; border-radius: 15px; text-align: center;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
            <h2 style='color: white;'>Przewidywany czas półmaratonu</h2>
            <h1 style='font-size: 72px; color: #FFD700;'>{godziny:02d}:{minuty:02d}:{sekundy:02d}</h1>
            <p style='font-size: 18px; color: #f0f0f0;'>({czas_sek:.0f} sekund)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Płeć", "👩 Kobieta" if pred['plec'] == 'K' else "👨 Mężczyzna")
        with col_b:
            st.metric("Tempo 5km", f"{pred['tempo']:.2f} min/km")
        with col_c:
            tempo_polmaraton = (czas_sek / 60) / 21.0975
            st.metric("Tempo półmaratonu", f"{tempo_polmaraton:.2f} min/km")

        st.markdown("---")
        st.subheader("📈 Analiza wyniku")

        if godziny < 1 or (godziny == 1 and minuty < 30):
            kategoria, opis = "🏆 Wyczynowy", "Gratulacje! To bardzo szybki czas!"
        elif godziny == 1 and minuty < 45:
            kategoria, opis = "⭐ Zaawansowany", "Świetny wynik!"
        elif godziny < 2:
            kategoria, opis = "👍 Średnio-zaawansowany", "Dobry czas, trenuj dalej!"
        else:
            kategoria, opis = "🎯 Początkujący", "Każdy start to sukces!"

        st.markdown(f"**Kategoria:** {kategoria}")
        st.info(opis)

    else:
        st.info("👈 Wprowadź swoje dane i kliknij 'Przewidź czas' aby zobaczyć wynik")
        st.markdown("### 💡 Przykładowe predykcje")
        st.markdown("""
        | Płeć | Tempo 5km | Przewidywany czas |
        |------|-----------|-------------------|
        | 👨 Mężczyzna | 5.0 min/km | ~1:45:00 |
        | 👩 Kobieta | 6.0 min/km | ~2:05:00 |
        | 👨 Mężczyzna | 7.0 min/km | ~2:25:00 |
        | 👩 Kobieta | 8.0 min/km | ~2:50:00 |
        """)

# ===============================
# STOPKA
# ===============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🏃 Aplikacja do przewidywania czasu półmaratonu | Oparta na uczeniu maszynowym</p>
    <p style='font-size: 12px;'>Model trenowany na rzeczywistych danych biegaczy. Wyniki mają charakter orientacyjny.</p>
</div>
""", unsafe_allow_html=True)
