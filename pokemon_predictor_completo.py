import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import requests
import os

# --- 1. CONFIGURACIÓN INICIAL Y CARGA DE RECURSOS ---

# Definir las rutas de los archivos
DATA_PATH = "pokemon.csv"
MODEL_PATH = "modelo_pokemon_SVM.pkl"
SCALER_PATH = "scaler_pokemon_SVM.pkl"

# Configuración de la página
st.set_page_config(
    page_title="Pokémon Battle Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Carga y preprocesa el DataFrame de Pokémon con estandarización de nombres."""
    try:
        # 1. Cargar el DataFrame
        df = pd.read_csv(DATA_PATH)
        
        # 🐛 CORRECCIÓN CLAVE: Estandarizar la columna 'name' a minúsculas y sin espacios
        df['name'] = df['name'].str.strip().str.lower()
        
        # Columnas necesarias para el modelo
        cols_base = [
            'name', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
            'height_m', 'weight_kg', 'is_legendary', 'type1', 'type2'
        ]
        cols_against = [c for c in df.columns if c.startswith('against_')]
        df = df[cols_base + cols_against]

        # --- IMPUTACIÓN DE NaN EN PESO Y ALTURA (MANTENIENDO LÓGICA ORIGINAL) ---
        stats_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'height_m', 'weight_kg']
        
        for col in stats_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
        df = df.dropna(subset=['type1'])
        df['type2'] = df['type2'].fillna('None')
        df['is_legendary'] = df['is_legendary'].astype(int)

        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos '{DATA_PATH}'.")
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
# 🐛 CORRECCIÓN: Usamos str.title() solo para la visualización en el selectbox
POKEMON_LIST_DISPLAY = sorted(df['name'].str.title().unique())


# --- 2. FUNCIONES DEL NÚCLEO DE PREDICCIÓN (Adaptadas) ---

def get_pokemon_row(pokemon_name):
    """Obtiene la fila del DataFrame usando el nombre en minúsculas."""
    # 🐛 CORRECCIÓN: Se asegura que la búsqueda sea con el nombre en minúsculas
    return df[df['name'] == pokemon_name.lower()].iloc[0]
def display_pokemon_card(col, name, prob_win=None, is_winner=False):
    """
    Muestra una tarjeta de Pokémon con imagen y estadísticas en un recuadro.
    """
    if name is None:
        col.empty()
        return

    # 1. Obtener datos del Pokémon
    row = get_pokemon_row(name)
    image_url = get_pokemon_image_url(name)
    
    # 2. Definir estilos
    color_border = "#4ade80" if is_winner else ("#f87171" if prob_win is not None and not is_winner else "#60a5fa") 
    color_text = "#16a34a" if is_winner else ("#dc2626" if prob_win is not None and not is_winner else "#2563eb")
    
    # --- 3. Generar el HTML para las Estadísticas ---
    # Colores base para las estadísticas (Estilo más neutral/oscuro)
    stats_html = f"""
    <div style="
        width: 100%; 
        background-color: #2e3034; /* Fondo oscuro para el recuadro */
        border-radius: 8px; 
        padding: 8px; 
        margin-top: 10px;
        color: #f0f0f0; /* Texto claro */
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
    ">
        <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
            <small>HP:</small> <strong>{int(row['hp'])}</strong>
            <small>Attack:</small> <strong>{int(row['attack'])}</strong>
            <small>Defense:</small> <strong>{int(row['defense'])}</strong>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <small>Sp.Atk:</small> <strong>{int(row['sp_attack'])}</strong>
            <small>Sp.Def:</small> <strong>{int(row['sp_defense'])}</strong>
            <small>Speed:</small> <strong>{int(row['speed'])}</strong>
        </div>
        <div style="text-align: center; margin-top: 5px; padding-top: 5px; border-top: 1px solid #444;">
            <small>TOTAL STATS:</small> <strong>{int(row['hp'] + row['attack'] + row['defense'] + row['sp_attack'] + row['sp_defense'] + row['speed'])}</strong>
        </div>
    </div>
    """
    
    # --- 4. HTML de la Tarjeta Completa ---
    card_html = f"""
    <div style="
        border: 4px solid {color_border};
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        height: 450px; /* Aumentamos la altura para acomodar las stats */
        display: flex;
        flex-direction: column;
        justify-content: start; /* Alineamos arriba */
        align-items: center;
        transition: border-color 0.3s ease;
    ">
        <h3 style="color: {color_text}; margin-bottom: 5px;">{name.title()}</h3>
        <img src="{image_url}" onerror="this.onerror=null; this.src='https://placehold.co/150x150/f0f0f0/888888?text=NO+IMAGE';" width="150" height="150" style="margin-bottom: 10px;">
        
        {stats_html}
        
    </div>
    """
    # Usamos la columna proporcionada para renderizar el HTML
    col.markdown(card_html, unsafe_allow_html=True)

def calcular_ventaja_completa(poke_a_name, poke_b_name):
    """Calcula la ventaja de tipo completa del Pokémon A contra el Pokémon B."""
    row_a = get_pokemon_row(poke_a_name)
    row_b = get_pokemon_row(poke_b_name)
    
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
    row1 = get_pokemon_row(poke1_name)
    row2 = get_pokemon_row(poke2_name)
    data = {}
    stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    
    # Diferencias y Ratios de Stats Base
    for stat in stats:
        data[f'diff_{stat}'] = row1[stat] - row2[stat]
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
        
        if pred == 1:
            ganador = pokemon1
            confianza = prob[1] 
            perdedor = pokemon2
        else:
            ganador = pokemon2
            confianza = prob[0]
            perdedor = pokemon1
        
        return ganador, confianza, perdedor, 1 - confianza

    except IndexError:
        st.error(f"Error: Uno de los Pokémon seleccionados no se encuentra en el DataFrame estandarizado. Intenta reiniciar la aplicación.")
        return None, 0, None, 0
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
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


# --- 4. INTERFAZ DE STREAMLIT ---

# Inicializar estado de sesión para controlar el combate
if 'confronted' not in st.session_state:
    st.session_state['confronted'] = False

def set_confronted_true():
    """Callback para establecer el estado de 'confronted' en True."""
    st.session_state['confronted'] = True
    
def reset_confronted():
    """Resetea el estado cuando cambian los Pokémon seleccionados."""
    st.session_state['confronted'] = False

# --- INTERFAZ PRINCIPAL ---
st.title("🏆 Pokémon Battle Predictor (SVM)")
st.markdown("Selecciona dos Pokémon para predecir el ganador de un combate, basado en un modelo de Machine Learning (SVM) entrenado con simulaciones de batalla.")

# Contenedor para la selección de Pokémon (usa dos columnas)
col1_select, col2_select = st.columns(2)

# Valores iniciales
default_poke1 = 'Pikachu'
default_poke2 = 'Charizard'

with col1_select:
    pokemon1_name = st.selectbox(
        "Pokémon 1",
        options=POKEMON_LIST_DISPLAY, # Usamos la lista de visualización
        index=POKEMON_LIST_DISPLAY.index(default_poke1) if default_poke1 in POKEMON_LIST_DISPLAY else 0,
        key="poke1",
        on_change=reset_confronted
    )

with col2_select:
    pokemon2_name = st.selectbox(
        "Pokémon 2",
        options=POKEMON_LIST_DISPLAY, # Usamos la lista de visualización
        index=POKEMON_LIST_DISPLAY.index(default_poke2) if default_poke2 in POKEMON_LIST_DISPLAY else 1,
        key="poke2",
        on_change=reset_confronted
    )

st.markdown("---")

# Botón para iniciar el combate en el centro
col_btn_left, col_btn_center, col_btn_right = st.columns([1, 1, 1])
with col_btn_center:
    st.button(
        "🔥 ¡CONFRONTAR! 🔥", 
        use_container_width=True, 
        type="primary",
        on_click=set_confronted_true
    )

st.markdown("---")

# Definir las columnas de visualización una sola vez
col1_display, col_center, col2_display = st.columns([1, 1, 1])


# Lógica de visualización y predicción
if st.session_state.confronted and pokemon1_name != pokemon2_name:
    
    # Obtener predicción (se llama con los nombres Title-case del selectbox)
    ganador, prob_ganador, perdedor, prob_perdedor = predecir_batalla(pokemon1_name, pokemon2_name, MODEL, SCALER)
    
    if ganador:
        st.balloons()
        # Columna Central: Ganador y Progreso
        with col_center:
            st.markdown("<h1 style='text-align: center; color: #16a34a;'>🏆 Ganador 🏆</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>{ganador.title()}</h2>", unsafe_allow_html=True)
            
            # Barra de progreso visual
            st.markdown("### Probabilidad de Victoria")
            
            if ganador == pokemon1_name:
                prob_poke1 = prob_ganador
                prob_poke2 = prob_perdedor
            else: 
                prob_poke1 = prob_perdedor
                prob_poke2 = prob_ganador
                
            st.progress(prob_poke1, text=f"{pokemon1_name.title()} ({prob_poke1:.1%}) vs {pokemon2_name.title()} ({prob_poke2:.1%})")

        # Mostrar tarjetas con colores de ganador/perdedor
        if ganador == pokemon1_name:
            with col1_display:
                display_pokemon_card(col1_display, pokemon1_name, prob_ganador, is_winner=True)
            with col2_display:
                display_pokemon_card(col2_display, pokemon2_name, prob_perdedor, is_winner=False)
        else: # Ganador es Pokemon 2
            with col1_display:
                display_pokemon_card(col1_display, pokemon1_name, prob_perdedor, is_winner=False)
            with col2_display:
                display_pokemon_card(col2_display, pokemon2_name, prob_ganador, is_winner=True)
        
    else:
        # En caso de error de predicción (ya se mostró el error en predecir_batalla)
        with col_center:
             st.error("No se pudo predecir la batalla. Verifica los datos o el modelo.")

elif pokemon1_name == pokemon2_name:
    # Caso 2: Nombres iguales (No Confrontado O Confrontado)
    st.warning("Por favor, selecciona dos Pokémon diferentes para la batalla.")
    st.session_state.confronted = False
    
    # Mostrar tarjetas neutrales
    with col1_display:
        display_pokemon_card(col1_display, pokemon1_name, prob_win=None, is_winner=False)
    with col_center:
        st.markdown("<h1 style='text-align: center; margin-top: 100px;'>VS</h1>", unsafe_allow_html=True)
    with col2_display:
        display_pokemon_card(col2_display, pokemon2_name, prob_win=None, is_winner=False)

else:
    # Caso 3: Estado inicial o Selección cambiada (No Confrontado)
    with col_center:
        st.markdown("<h1 style='text-align: center; margin-top: 100px;'>VS</h1>", unsafe_allow_html=True)
        st.info("Presiona ¡CONFRONTAR! para ver la predicción.")
    
    # Mostrar tarjetas neutrales
    with col1_display:
        display_pokemon_card(col1_display, pokemon1_name, prob_win=None, is_winner=False)
    with col2_display:
        display_pokemon_card(col2_display, pokemon2_name, prob_win=None, is_winner=False)