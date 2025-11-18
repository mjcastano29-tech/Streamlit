# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import requests
import os

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE RECURSOS ---

# Configuración de la página
st.set_page_config(
    page_title="Pokémon Battle Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Definir las rutas de los archivos (Asegúrate de que estos archivos existan)
DATA_PATH = "pokemon.csv"
MODEL_PATH = "modelo_pokemon_SVM.pkl"
SCALER_PATH = "scaler_pokemon_SVM.pkl"

@st.cache_data
def load_data():
    """Carga y preprocesa el DataFrame de Pokémon."""
    try:
        # Nota: Asegúrate de que 'pokemon.csv' esté en el mismo directorio que este script.
        df = pd.read_csv(DATA_PATH)
        
        # Columnas necesarias para el modelo
        cols_base = [
            'name', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
            'height_m', 'weight_kg', 'is_legendary', 'type1', 'type2'
        ]
        cols_against = [c for c in df.columns if c.startswith('against_')]
        df = df[cols_base + cols_against]

        df = df.dropna(subset=['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'type1'])
        df['type2'] = df['type2'].fillna('None')
        df['is_legendary'] = df['is_legendary'].astype(int)
        
        # Limpieza de nombres para que coincidan con la API
        df['name'] = df['name'].str.lower()

        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos '{DATA_PATH}'. Por favor, colócalo en el mismo directorio.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar o procesar los datos: {e}")
        st.stop()


@st.cache_resource
def load_model_and_scaler():
    """Carga el modelo y el escalador entrenados."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: No se encontraron los archivos del modelo ('{MODEL_PATH}' o '{SCALER_PATH}').")
        st.info("Asegúrate de haber ejecutado previamente tu script de entrenamiento para generar estos archivos.")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar el modelo o el escalador: {e}")
        st.stop()

# Cargar datos y modelo
df = load_data()
MODEL, SCALER = load_model_and_scaler()
POKEMON_LIST = sorted(df['name'].unique())


# --- 2. FUNCIONES DEL NÚCLEO DE PREDICCIÓN (Adaptadas del código del usuario) ---

def calcular_ventaja_completa(poke_a_name, poke_b_name):
    """Calcula la ventaja de tipo completa del Pokémon A contra el Pokémon B."""
    row_a = df[df['name'] == poke_a_name].iloc[0]
    row_b = df[df['name'] == poke_b_name].iloc[0]
    
    tipos_a = [row_a['type1']]
    if row_a['type2'] != 'None':
        tipos_a.append(row_a['type2'])
    
    ventaja_total = 1.0
    for tipo_a in tipos_a:
        col_against = f"against_{tipo_a.lower()}"
        if col_against in row_b:
            # Multiplica por el valor de resistencia/debilidad
            ventaja_total *= row_b[col_against]
            
    return ventaja_total

def crear_features_mejoradas(poke1_name, poke2_name):
    """Crea el vector de features para la predicción del modelo."""
    row1 = df[df['name'] == poke1_name].iloc[0]
    row2 = df[df['name'] == poke2_name].iloc[0]
    data = {}
    stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    
    # Diferencias y Ratios de Stats Base
    for stat in stats:
        data[f'diff_{stat}'] = row1[stat] - row2[stat]
        # Usamos max(row2[stat], 1) para evitar división por cero
        data[f'ratio_{stat}'] = row1[stat] / max(row2[stat], 1)

    # Stats Totales y Comparativos
    data['total_stats_1'] = sum(row1[stat] for stat in stats)
    data['total_stats_2'] = sum(row2[stat] for stat in stats)
    data['diff_total_stats'] = data['total_stats_1'] - data['total_stats_2']

    # Ventajas de tipo
    data['ventaja_tipo_1vs2'] = calcular_ventaja_completa(poke1_name, poke2_name)
    data['ventaja_tipo_2vs1'] = calcular_ventaja_completa(poke2_name, poke1_name)
    data['balance_ventaja_tipo'] = data['ventaja_tipo_1vs2'] - data['ventaja_tipo_2vs1']

    # Otros features
    data['diff_height'] = row1['height_m'] - row2['height_m']
    data['diff_weight'] = row1['weight_kg'] - row2['weight_kg']
    data['diff_legendary'] = row1['is_legendary'] - row2['is_legendary']
    
    return data

def predecir_batalla(pokemon1, pokemon2, model, scaler):
    """Genera la predicción del ganador y su probabilidad."""
    try:
        # 1. Crear features
        features = crear_features_mejoradas(pokemon1, pokemon2)
        features_df = pd.DataFrame([features])
        
        # 2. Escalar features
        features_scaled = scaler.transform(features_df)
        
        # 3. Predecir probabilidad y resultado
        prob = model.predict_proba(features_scaled)[0]
        pred = model.predict(features_scaled)[0]
        
        # pred == 1 significa que el Pokémon 1 (el primero en el orden) gana
        # pred == 0 significa que el Pokémon 2 (el segundo en el orden) gana
        
        if pred == 1:
            ganador = pokemon1
            confianza = prob[1] 
            perdedor = pokemon2
        else:
            ganador = pokemon2
            confianza = prob[0]
            perdedor = pokemon1
        
        # Retorna el nombre del ganador, la probabilidad de ganar, y la probabilidad del perdedor
        return ganador, confianza, perdedor, 1 - confianza

    except Exception as e:
        #st.error(f"Error en la predicción: {e}")
        return None, 0, None, 0

# --- 3. FUNCIONES DE INTERFAZ Y POKÉAPI ---

@st.cache_data(ttl=3600)
def get_pokemon_image_url(pokemon_name):
    """Obtiene la URL de la imagen del Pokémon desde la PokéAPI."""
    # La API requiere nombres en minúsculas y sin espacios
    name = pokemon_name.lower().replace(" ", "-") 
    url = f"https://pokeapi.co/api/v2/pokemon/{name}"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Lanza error para códigos 4xx/5xx
        data = response.json()
        
        # Usamos el sprite frontal del arte oficial
        official_artwork = data['sprites']['other']['official-artwork']['front_default']
        return official_artwork
    except requests.exceptions.RequestException:
        # Fallback si la API no encuentra el nombre (ej. variantes, formas especiales, o si el nombre no coincide)
        return "https://placehold.co/150x150/f0f0f0/888888?text=NO+IMAGE"

def display_pokemon_card(col, name, prob_win=None, is_winner=False):
    """Muestra una tarjeta de Pokémon con su imagen y estadísticas."""
    if name is None:
        col.empty()
        return

    poke_data = df[df['name'] == name].iloc[0]
    
    # Obtener imagen
    image_url = get_pokemon_image_url(name)
    
    # Estilo de la tarjeta
    color_border = "#4ade80" if is_winner else "#f87171"
    
    card_html = f"""
    <div style="
        border: 4px solid {color_border};
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: scale(1.0);
        transition: transform 0.2s;
    ">
        <h3 style="color: #262626; margin-bottom: 5px;">{name.title()}</h3>
        <img src="{image_url}" onerror="this.onerror=null; this.src='https://placehold.co/150x150/f0f0f0/888888?text=NO+IMAGE';" width="150" height="150" style="margin-bottom: 10px;">
        
        <table style="width: 100%; font-size: 14px; text-align: left; margin-bottom: 5px; border-collapse: collapse;">
            <tr><td style="padding: 2px 0;">HP:</td><td style="text-align: right;">{poke_data['hp']}</td></tr>
            <tr><td style="padding: 2px 0;">Attack:</td><td style="text-align: right;">{poke_data['attack']}</td></tr>
            <tr><td style="padding: 2px 0;">Defense:</td><td style="text-align: right;">{poke_data['defense']}</td></tr>
            <tr><td style="padding: 2px 0;">Speed:</td><td style="text-align: right;">{poke_data['speed']}</td></tr>
            <tr><td style="padding: 2px 0;">Type 1:</td><td style="text-align: right;">{poke_data['type1'].title()}</td></tr>
            <tr><td style="padding: 2px 0;">Legendary:</td><td style="text-align: right;">{'Yes' if poke_data['is_legendary'] == 1 else 'No'}</td></tr>
        </table>
        
        {'<h2 style="color: #16a34a; margin-top: 10px;">Win Prob: ' + f'{prob_win:.1%}' + '</h2>' if prob_win is not None else ''}
        
    </div>
    """
    col.markdown(card_html, unsafe_allow_html=True)


# --- 4. INTERFAZ DE STREAMLIT ---

st.title("🏆 Pokémon Battle Predictor (SVM)")
st.markdown("Selecciona dos Pokémon para predecir el ganador de un combate, basado en un modelo de Machine Learning (SVM) entrenado con simulaciones de batalla.")

# Contenedor para la selección de Pokémon (usa dos columnas)
col1_select, col2_select = st.columns(2)

with col1_select:
    pokemon1_name = st.selectbox(
        "Pokémon 1",
        options=POKEMON_LIST,
        index=POKEMON_LIST.index('pikachu') if 'pikachu' in POKEMON_LIST else 0,
        key="poke1"
    )

with col2_select:
    pokemon2_name = st.selectbox(
        "Pokémon 2",
        options=POKEMON_LIST,
        index=POKEMON_LIST.index('charizard') if 'charizard' in POKEMON_LIST else 1,
        key="poke2"
    )

st.markdown("---")


# Lógica de predicción si ambos están seleccionados y son diferentes
if pokemon1_name == pokemon2_name:
    st.warning("Por favor, selecciona dos Pokémon diferentes para la batalla.")
    # Muestra las dos cartas de forma neutral
    col1_display, col_vs, col2_display = st.columns([1, 0.5, 1])
    display_pokemon_card(col1_display, pokemon1_name)
    col_vs.markdown("<h1 style='text-align: center; margin-top: 100px;'>VS</h1>", unsafe_allow_html=True)
    display_pokemon_card(col2_display, pokemon2_name)
else:
    # Obtener predicción
    ganador, prob_ganador, perdedor, prob_perdedor = predecir_batalla(pokemon1_name, pokemon2_name, MODEL, SCALER)
    
    # ------------------
    # MOSTRAR RESULTADOS
    # ------------------
    
    col1_display, col_center, col2_display = st.columns([1, 1, 1])

    if ganador:
        st.subheader("Resultado de la Predicción")
        
        # Columna Central: Ganador y Progreso
        with col_center:
            st.markdown("<h1 style='text-align: center; color: #16a34a;'>🏆 Ganador 🏆</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{ganador.title()}</h2>", unsafe_allow_html=True)
            
            # Barra de progreso visual
            st.markdown("### Probabilidad de Victoria")
            if ganador == pokemon1_name:
                 st.progress(prob_ganador, text=f"{pokemon1_name.title()} ({prob_ganador:.1%}) vs {pokemon2_name.title()} ({prob_perdedor:.1%})")
                 
                 # Mostrar tarjetas con colores de ganador/perdedor
                 display_pokemon_card(col1_display, pokemon1_name, prob_ganador, is_winner=True)
                 display_pokemon_card(col2_display, pokemon2_name, prob_perdedor, is_winner=False)
                 
            else: # Ganador es Pokemon 2
                 st.progress(prob_perdedor, text=f"{pokemon1_name.title()} ({prob_perdedor:.1%}) vs {pokemon2_name.title()} ({prob_ganador:.1%})")

                 # Mostrar tarjetas con colores de ganador/perdedor
                 display_pokemon_card(col1_display, pokemon1_name, prob_perdedor, is_winner=False)
                 display_pokemon_card(col2_display, pokemon2_name, prob_ganador, is_winner=True)

        
    else:
        st.error("No se pudo predecir la batalla. Verifica los datos o el modelo.")


# Entrenamiento final con todos los datos
model.fit(X_scaled, y) 

# Guardar modelo y escalador
joblib.dump(model, "modelo_pokemon_SVM.pkl")
joblib.dump(scaler, "scaler_pokemon_SVM.pkl")