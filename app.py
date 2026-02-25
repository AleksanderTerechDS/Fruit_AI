import streamlit as st
import joblib
import numpy as np

# Konfiguracja strony
st.set_page_config(page_title="Fruit Expert AI", page_icon="🥝", layout="wide")

st.title("🍓 Inteligentny Ekspert Owocowy")
st.markdown("---")

# PASEK BOCZNY - Parametry
st.sidebar.header("Parametry Owocu")
waga = st.sidebar.slider("Waga (g)", 5, 500, 150)
tekstura = st.sidebar.radio("Tekstura", ["Gładka", "Szorstka"])
kolor_nazwa = st.sidebar.selectbox("Kolor", ["Zielony", "Czerwony", "Pomarańczowy", "Żółty", "Ciemny"])
ksztalt_nazwa = st.sidebar.selectbox("Kształt", ["Okrągły", "Podłużny"])

# Mapowanie na liczby
tekstura_num = 0 if tekstura == "Gładka" else 1
mapa_kolorow = {"Zielony": 0, "Czerwony": 1, "Pomarańczowy": 2, "Żółty": 3, "Ciemny": 4}
mapa_ksztaltow = {"Okrągły": 0, "Podłużny": 1}

# LOGIKA GŁÓWNA
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Twoje wejście")
    st.info(f"Waga: {waga}g | Kolor: {kolor_nazwa} | Kształt: {ksztalt_nazwa}")

    if st.button("Analizuj Owoc", use_container_width=True):
        model = joblib.load('model_owocow.pkl')
        dane = np.array([[waga, tekstura_num, mapa_kolorow[kolor_nazwa], mapa_ksztaltow[ksztalt_nazwa]]])

        wynik = model.predict(dane)[0]
        proby = model.predict_proba(dane)[0]
        pewnosc = max(proby) * 100

        with col2:
            st.subheader("Werdykt AI")
            st.success(f"To prawdopodobnie: **{wynik.upper()}**")
            st.metric("Pewność", f"{pewnosc:.1f}%")

            # Dynamiczne zdjęcie (pamiętaj o plikach!)
            try:
                st.image(f"{wynik}.jpg", width=300)
            except:
                st.warning("Brak zdjęcia w bazie.")

            if pewnosc < 50:
                st.warning("AI nie jest pewne! Spróbuj podać dokładniejsze dane.")
            else:
                st.balloons()