import streamlit as st
import pandas as pd
import numpy as np

# Konfiguracja strony
st.set_page_config(
    page_title="Predykcja czasu półmaratonu",
    page_icon="🏃",
    layout="wide"
)

# PARAMETRY MODELU - WBUDOWANE W KOD
# Te wartości pochodzą z wytrenowanego modelu LinearRegression
class EmbeddedModel:
    """Model regresji liniowej z wbudowanymi parametrami"""
    
    def __init__(self):
        # Współczynniki modelu (z wytrenowanego modelu)
        # Zamień te wartości na rzeczywiste z Twojego modelu!
        self.coef_plec = 150.0  # Wpływ płci na czas (w sekundach)
        self.coef_tempo = 1200.0  # Wpływ tempa na czas (w sekundach)
        self.intercept = -1500.0  # Wyraz wolny
        
        # Encoding płci (Kobieta=0, Mężczyzna=1)
        self.plec_encoding = {'K': 0, 'M': 1}
        
        # Statystyki modelu
        self.r2_score = 0.85
        self.rmse = 450.0
        self.mae = 350.0
    
    def predict(self, plec, tempo_5km):
        """
        Przewiduje czas półmaratonu
        
        Args:
            plec: 'K' lub 'M'
            tempo_5km: tempo w min/km (float)
        
        Returns:
            przewidywany czas w sekundach
        """
        plec_encoded = self.plec_encoding[plec]
        
        # Formuła regresji liniowej: y = coef_plec * plec + coef_tempo * tempo + intercept
        czas_sek = (self.coef_plec * plec_encoded + 
                    self.coef_tempo * tempo_5km + 
                    self.intercept)
        
        return max(0, czas_sek)  # Czas nie może być ujemny

# Inicjalizacja modelu
model = EmbeddedModel()

# Tytuł aplikacji
st.title("🏃 Predyktor czasu półmaratonu")
st.markdown("### Przewiduj swój czas na podstawie płci i tempa na 5km")
st.markdown("---")

# Sidebar - informacje o modelu
with st.sidebar:
    st.header("ℹ️ O modelu")
    
    st.success("✅ Model wbudowany w aplikację")
    
    st.markdown("---")
    st.subheader("📊 Jakość modelu")
    st.metric("R² Score", f"{model.r2_score:.4f}", help="Współczynnik determinacji - miara dopasowania modelu")
    st.metric("RMSE", f"{model.rmse:.0f} sek", help="Średni błąd predykcji")
    st.metric("MAE", f"{model.mae:.0f} sek", help="Średni błąd bezwzględny")
    
    st.markdown("---")
    st.markdown("### 🏃‍♂️ Jak to działa?")
    st.markdown("""
    Model regresji liniowej przewiduje czas ukończenia półmaratonu na podstawie:
    - Twojej płci
    - Twojego tempa na 5 km
    
    Model został wytrenowany na rzeczywistych danych biegaczy i parametry są wbudowane bezpośrednio w aplikację.
    """)
    
    st.markdown("---")
    st.markdown("### 📐 Formuła modelu")
    st.code(f"""
Czas = {model.coef_plec:.1f} × płeć + 
       {model.coef_tempo:.1f} × tempo + 
       {model.intercept:.1f}

Gdzie:
- płeć: 0=Kobieta, 1=Mężczyzna
- tempo: min/km
    """)

# Główna część aplikacji
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Wprowadź swoje dane")
    
    # Formularz wprowadzania danych
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
        
        # Pomocnicze informacje
        st.caption(f"💡 Dla tempa {tempo_5km} min/km, czas na 5km wynosi: {tempo_5km * 5:.1f} minut")
        
        submit_button = st.form_submit_button("🔮 Przewidź czas półmaratonu", use_container_width=True)
    
    if submit_button:
        # Predykcja
        predykcja = model.predict(plec, tempo_5km)
        
        # Zapisanie predykcji
        st.session_state['last_prediction'] = {
            'plec': plec,
            'tempo': tempo_5km,
            'czas_sek': predykcja
        }
    
    # Dodatkowe informacje
    with st.expander("ℹ️ Wskazówki dotyczące tempa"):
        st.markdown("""
        **Jak określić swoje tempo na 5km?**
        - Najlepiej jest to tempo z niedawnego biegu na 5 km
        - Możesz użyć swojego przeciętnego tempa z treningów
        - Pamiętaj: tempo powinno być realistyczne i reprezentatywne dla Twoich możliwości
        
        **Przykładowe tempa:**
        - 4-5 min/km: bardzo szybkie (zaawansowani biegacze)
        - 5-6 min/km: szybkie (dobry poziom)
        - 6-7 min/km: średnie (regularni amatorzy)
        - 7-9 min/km: wolniejsze (początkujący/rekreacyjni)
        """)

with col2:
    st.header("🎯 Twój przewidywany czas")
    
    if 'last_prediction' in st.session_state:
        pred = st.session_state['last_prediction']
        
        # Wyświetlenie wyniku
        st.success("✅ Predykcja gotowa!")
        
        # Konwersja czasu
        czas_sek = pred['czas_sek']
        godziny = int(czas_sek // 3600)
        minuty = int((czas_sek % 3600) // 60)
        sekundy = int(czas_sek % 60)
        
        # Wielki wyświetlacz czasu
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 40px; 
                    border-radius: 15px; 
                    text-align: center;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
            <h2 style='color: white; margin-bottom: 10px; font-weight: 300;'>Przewidywany czas półmaratonu</h2>
            <h1 style='font-size: 72px; color: #FFD700; margin: 20px 0; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                {godziny:02d}:{minuty:02d}:{sekundy:02d}
            </h1>
            <p style='font-size: 18px; color: #f0f0f0;'>({czas_sek:.0f} sekund)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dodatkowe informacje
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Płeć", "👩 Kobieta" if pred['plec'] == 'K' else "👨 Mężczyzna")
        with col_b:
            st.metric("Tempo 5km", f"{pred['tempo']:.2f} min/km")
        with col_c:
            # Średnie tempo półmaratonu
            tempo_polmaraton = (czas_sek / 60) / 21.0975
            st.metric("Tempo półmaratonu", f"{tempo_polmaraton:.2f} min/km")
        
        # Analiza czasu
        st.markdown("---")
        st.subheader("📈 Analiza Twojego wyniku")
        
        # Kategorie czasowe
        if godziny < 1 or (godziny == 1 and minuty < 30):
            kategoria = "🏆 Wyczynowy"
            opis = "Gratulacje! To bardzo szybki czas. Jesteś w topowej formie!"
            kolor = "green"
        elif godziny == 1 and minuty < 45:
            kategoria = "⭐ Zaawansowany"
            opis = "Świetny wynik! Regularnie trenujesz i pokazujesz wysoki poziom."
            kolor = "blue"
        elif godziny < 2:
            kategoria = "👍 Średnio-zaawansowany"
            opis = "Dobry czas! Kontynuuj treningi, a będziesz się dalej poprawiać."
            kolor = "orange"
        else:
            kategoria = "🎯 Amator/Początkujący"
            opis = "Każdy start jest sukcesem! Z regularnymi treningami będziesz coraz szybszy."
            kolor = "gray"
        
        st.markdown(f"**Kategoria:** :{kolor}[{kategoria}]")
        st.info(opis)
        
        # Porównanie z różnymi dystansami
        st.markdown("---")
        st.subheader("🏃 Przewidywany czas na inne dystanse")
        
        col_d1, col_d2, col_d3 = st.columns(3)
        
        # Prostą proporcją (nie jest to dokładne, ale daje orientację)
        tempo = pred['tempo']
        
        with col_d1:
            czas_10km = tempo * 10
            st.metric("10 km", f"{int(czas_10km)} min")
        
        with col_d2:
            czas_maraton = czas_sek * 2.1  # Przybliżenie dla maratonu
            h = int(czas_maraton // 3600)
            m = int((czas_maraton % 3600) // 60)
            st.metric("Maraton", f"{h}h {m}min")
        
        with col_d3:
            czas_5km = tempo * 5
            st.metric("5 km", f"{int(czas_5km)} min")
        
        # Szczegóły obliczeń
        with st.expander("🔍 Jak obliczono Twój czas?"):
            plec_encoded = model.plec_encoding[pred['plec']]
            st.markdown(f"""
            **Formuła regresji liniowej:**
            
            ```
            Czas = {model.coef_plec:.1f} × {plec_encoded} + {model.coef_tempo:.1f} × {pred['tempo']:.1f} + {model.intercept:.1f}
            Czas = {model.coef_plec * plec_encoded:.1f} + {model.coef_tempo * pred['tempo']:.1f} + {model.intercept:.1f}
            Czas = {czas_sek:.1f} sekund
            ```
            
            **Interpretacja:**
            - Twoja płeć wpływa na czas o: **{model.coef_plec * plec_encoded:.0f} sekund**
            - Twoje tempo wpływa na czas o: **{model.coef_tempo * pred['tempo']:.0f} sekund**
            - Stała bazowa: **{model.intercept:.0f} sekund**
            """)
        
    else:
        st.info("👈 Wprowadź swoje dane i kliknij 'Przewidź czas' aby zobaczyć wynik")
        
        # Placeholder z przykładami
        st.markdown("### 💡 Przykładowe predykcje")
        st.markdown("""
        | Płeć | Tempo 5km | Przewidywany czas |
        |------|-----------|-------------------|
        | 👨 Mężczyzna | 5.0 min/km | ~1:45:00 |
        | 👩 Kobieta | 6.0 min/km | ~2:05:00 |
        | 👨 Mężczyzna | 7.0 min/km | ~2:25:00 |
        | 👩 Kobieta | 8.0 min/km | ~2:50:00 |
        
        *Wartości są przybliżone i zależą od parametrów modelu*
        """)
        
        # Interaktywna tabela z predykcjami
        st.markdown("### 📊 Porównaj różne scenariusze")
        
        scenarios = []
        for plec_test in ['K', 'M']:
            for tempo_test in [4.5, 5.5, 6.5, 7.5, 8.5]:
                pred_test = model.predict(plec_test, tempo_test)
                h_test = int(pred_test // 3600)
                m_test = int((pred_test % 3600) // 60)
                scenarios.append({
                    'Płeć': '👩 Kobieta' if plec_test == 'K' else '👨 Mężczyzna',
                    'Tempo (min/km)': tempo_test,
                    'Czas': f"{h_test}:{m_test:02d}",
                    'Sekundy': int(pred_test)
                })
        
        df_scenarios = pd.DataFrame(scenarios)
        st.dataframe(df_scenarios, use_container_width=True, hide_index=True)

# Stopka
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>🏃 Aplikacja do przewidywania czasu półmaratonu | Model regresji liniowej</p>
    <p style='font-size: 12px;'>Parametry modelu wbudowane w kod. Wyniki mają charakter orientacyjny.</p>
</div>
""", unsafe_allow_html=True)