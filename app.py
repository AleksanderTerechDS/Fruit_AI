import streamlit as st
import joblib
import numpy as np
import os

# Konfiguracja strony
st.set_page_config(page_title="Fruit Expert AI v2.0", page_icon="🥝", layout="wide")

st.title("🍓 Inteligentny Ekspert Owocowy v2.0")
st.markdown("---")

# PASEK BOCZNY - Parametry
st.sidebar.header("⚙️ Parametry Owocu")
waga = st.sidebar.slider("Waga (g)", 3, 500, 150)
tekstura = st.sidebar.radio("Tekstura", ["Gładka", "Szorstka"])
kolor_nazwa = st.sidebar.selectbox("Kolor", ["Zielony", "Czerwony", "Pomarańczowy", "Żółty", "Ciemny"])
ksztalt_nazwa = st.sidebar.selectbox("Kształt", ["Okrągły", "Podłużny"])

# Mapowanie na liczby (musi być identyczne jak w train.py!)
mapa_tekstury = {"Gładka": 0, "Szorstka": 1}
mapa_kolorow = {"Zielony": 0, "Czerwony": 1, "Pomarańczowy": 2, "Żółty": 3, "Ciemny": 4}
mapa_ksztaltow = {"Okrągły": 0, "Podłużny": 1}

# LOGIKA GŁÓWNA
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🔍 Dane wejściowe")
    st.info(f"⚖️ Waga: {waga}g | 🎨 Kolor: {kolor_nazwa} | 📐 Kształt: {ksztalt_nazwa}")

    if st.button("ANALIZUJ OWOC", use_container_width=True):
        try:
            # Ładowanie modelu
            model = joblib.load('model_owocow.pkl')

            # Przygotowanie danych do predykcji
            dane = np.array([[waga, mapa_tekstury[tekstura], mapa_kolorow[kolor_nazwa], mapa_ksztaltow[ksztalt_nazwa]]])

            # Predykcja
            wynik = model.predict(dane)[0]
            proby = model.predict_proba(dane)[0]
            klasy = model.classes_
            pewnosc_max = max(proby) * 100

            with col2:
                st.subheader("🤖 Wynik Analizy AI")

                # Główny werdykt
                st.success(f"Werdykt: **{wynik.upper()}**")
                st.metric("Pewność najwyższa", f"{pewnosc_max:.1f}%")

                # Sekcja Prawdopodobieństwa (Paski postępu)
                st.markdown("---")
                st.write("📊 **Szczegółowy rozkład prawdopodobieństwa:**")

                for i in range(len(klasy)):
                    procent = proby[i] * 100
                    nazwa_owocu = klasy[i].capitalize()

                    # Wyświetlamy tylko te owoce, które mają chociaż 1% szans
                    if procent > 1:
                        st.write(f"{nazwa_owocu}: {procent:.1f}%")
                        st.progress(proby[i])

                # Wyświetlanie zdjęcia
                st.markdown("---")
                sciezka_foto = f"{wynik}.jpg"
                if os.path.exists(sciezka_foto):
                    st.image(sciezka_foto, width=350, caption=f"To jest {wynik}")
                else:
                    st.warning(f"Brak pliku: {sciezka_foto}")

                if pewnosc_max > 80:
                    st.balloons()
                elif pewnosc_max < 50:
                    st.warning("⚠️ Model ma wątpliwości. Spróbuj zmienić parametry!")

        except Exception as e:
            st.error(f"Wystąpił błąd: {e}")
            st.info("Upewnij się, że plik 'model_owocow.pkl' istnieje i został poprawnie wytrenowany.")

else:
with col2:
    st.write("👈 Skonfiguruj owoc w panelu bocznym i kliknij przycisk, aby zobaczyć magię AI!")