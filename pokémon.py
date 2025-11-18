import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score

import joblib



"""### Procesamiento"""



# Cargar y preparar datos

df = pd.read_csv("C:/Users/Salad507.UEXTERNADO/Downloads/pokemon.csv")



cols_base = [

    'name', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',

    'height_m', 'weight_kg', 'is_legendary', 'type1', 'type2'

]

cols_against = [c for c in df.columns if c.startswith('against_')]

df = df[cols_base + cols_against]



df = df.dropna(subset=['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'type1'])

df['type2'] = df['type2'].fillna('None')

df['is_legendary'] = df['is_legendary'].astype(int)

df



def aplicar_ataque(atacante, defensor, hp_defensor, turno):

    if atacante['attack'] >= atacante['sp_attack']: #Elegir tipo de ataque (físico o especial)

        poder_ataque = atacante['attack']

        poder_defensa = defensor['defense']

        tipo_ataque = atacante['type1']

    else:

        poder_ataque = atacante['sp_attack']

        poder_defensa = defensor['sp_defense']

        tipo_ataque = atacante['type2'] if atacante['type2'] != 'None' else atacante['type1']



    efectividad = calcular_efectividad_movimiento(tipo_ataque, defensor) #Calcular efectividad del movimiento

    daño_base = (poder_ataque * 0.5) - (poder_defensa * 0.25) #Calcular daño base

    daño_base = max(daño_base, 5)

    daño_final = daño_base * efectividad * np.random.uniform(0.85, 1.0) #Aplicar efectividad y variación aleatoria

    nuevo_hp = hp_defensor - daño_final #Calcular nuevo HP

    return max(nuevo_hp, 0)





def calcular_efectividad_movimiento(tipo_movimiento, defensor):

    tipos_defensor = [defensor['type1']]

    if defensor['type2'] != 'None':

        tipos_defensor.append(defensor['type2'])

    efectividad_total = 1.0

    for tipo_def in tipos_defensor:

        col_against = f"against_{tipo_movimiento.lower()}"

        if col_against in defensor:

            efectividad_total *= defensor[col_against] #Calcular efectividad contra cada tipo

    return efectividad_total



def simular_batalla_real(poke1, poke2):

    row1 = df[df['name'] == poke1].iloc[0]

    row2 = df[df['name'] == poke2].iloc[0]

    hp1, hp2 = row1['hp'], row2['hp']

    poke1_primero = row1['speed'] >= row2['speed'] #Determinar quién ataca primero



    for _ in range(10): #Bucle de turnos (máximo 10 turnos)

        if hp1 <= 0 or hp2 <= 0:

            break

        if poke1_primero: # Lógica de turnos - Caso 1: Pokémon 1 ataca primero

            hp2 = aplicar_ataque(row1, row2, hp2, _)

            if hp2 > 0:

                hp1 = aplicar_ataque(row2, row1, hp1, _)

        else:

            hp1 = aplicar_ataque(row2, row1, hp1, _)

            if hp1 > 0:

                hp2 = aplicar_ataque(row1, row2, hp2, _)



    if hp1 <= 0 and hp2 <= 0:

        return 1 if row1['hp'] >= row2['hp'] else 0

    elif hp1 <= 0:

        return 0

    else:

        return 1 #Determinar el ganador después de la batalla



def generar_batallas_estrategicas(n_batallas=2000):

    batallas = []

    pokemons = df['name'].values

    tipos_unicos = df['type1'].unique()



    for _ in range(n_batallas // 2):

        p1, p2 = np.random.choice(pokemons, 2, replace=False)

        batallas.append((p1, p2))



    for _ in range(n_batallas // 4):

        tipo_fuerte = np.random.choice(tipos_unicos)

        tipo_debil = np.random.choice(tipos_unicos)

        poke_fuerte = df[df['type1'] == tipo_fuerte]['name'].sample(1).iloc[0]

        poke_debil = df[df['type1'] == tipo_debil]['name'].sample(1).iloc[0]

        batallas.append((poke_fuerte, poke_debil))



    for _ in range(n_batallas // 4):

        legendarios = df[df['is_legendary'] == 1]['name']

        no_legendarios = df[df['is_legendary'] == 0]['name']

        if len(legendarios) > 0 and len(no_legendarios) > 0:

            l = legendarios.sample(1).iloc[0]

            n = no_legendarios.sample(1).iloc[0]

            batallas.append((l, n))

    return batallas



def calcular_ventaja_completa(poke_a, poke_b): #Considera todos los tipos de A contra todos los tipos de B

    tipos_a = [poke_a['type1']]

    if poke_a['type2'] != 'None':

        tipos_a.append(poke_a['type2'])

    tipos_b = [poke_b['type1']]

    if poke_b['type2'] != 'None':

        tipos_b.append(poke_b['type2'])

    ventaja_total = 1.0

    for tipo_a in tipos_a:

        for tipo_b in tipos_b:

            col_against = f"against_{tipo_a.lower()}"

            if col_against in poke_b:

                ventaja_total *= poke_b[col_against]

    return ventaja_total



def crear_features_mejoradas(poke1, poke2):

    row1 = df[df['name'] == poke1].iloc[0]

    row2 = df[df['name'] == poke2].iloc[0]

    data = {}

    stats = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

#Diferencias y Ratios de Stats Base

    for stat in stats:

        data[f'diff_{stat}'] = row1[stat] - row2[stat]

        data[f'ratio_{stat}'] = row1[stat] / max(row2[stat], 1)

#Stats Totales y Comparativos

    data['total_stats_1'] = sum(row1[stat] for stat in stats)

    data['total_stats_2'] = sum(row2[stat] for stat in stats)

    data['diff_total_stats'] = data['total_stats_1'] - data['total_stats_2']



    data['ventaja_tipo_1vs2'] = calcular_ventaja_completa(row1, row2)

    data['ventaja_tipo_2vs1'] = calcular_ventaja_completa(row2, row1)

    data['balance_ventaja_tipo'] = data['ventaja_tipo_1vs2'] - data['ventaja_tipo_2vs1']



    data['diff_height'] = row1['height_m'] - row2['height_m']

    data['diff_weight'] = row1['weight_kg'] - row2['weight_kg']

    data['diff_legendary'] = row1['is_legendary'] - row2['is_legendary']

    return data



#Hacer Batallas

print(" Generando batallas estratégicas...")

batallas = generar_batallas_estrategicas(2000)



print("Simulando batallas realistas...")

batallas_features = []

for i, (p1, p2) in enumerate(batallas):

    if i % 500 == 0:

        print(f"Procesando batalla {i}/{len(batallas)}")

    features = crear_features_mejoradas(p1, p2)

    features['winner'] = simular_batalla_real(p1, p2)

    batallas_features.append(features)



batallas_features_df = pd.DataFrame(batallas_features)

batallas_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

batallas_features_df.dropna(inplace=True)



print(f"Dataset generado: {batallas_features_df.shape[0]} batallas")



#SVM+K-FOLDS

X = batallas_features_df.drop(columns=['winner'])

y = batallas_features_df['winner']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



# Modelo SVM

model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)



# Validación cruzada estratificada

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')

f1_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='f1')



print("\nResultados del modelo SVM:")

print(f"Accuracy promedio: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")

print(f"F1-score promedio: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")



# Entrenamiento final con todos los datos

model.fit(X_scaled, y)



# Evaluación global

y_pred = model.predict(X_scaled)

acc = accuracy_score(y, y_pred)

f1 = f1_score(y, y_pred)

print(f"\n Entrenamiento final completado")

print(f"Accuracy global: {acc:.3f}")

print(f"F1-score global: {f1:.3f}")



# Guardar modelo y escalador

joblib.dump(model, "modelo_pokemon_SVM.pkl")

joblib.dump(scaler, "scaler_pokemon_SVM.pkl")



print("\n Modelo SVM guardado exitosamente.")



#Nuevas Batallas

def predecir_batalla(pokemon1, pokemon2):

    try:

        features = crear_features_mejoradas(pokemon1, pokemon2)

        features_df = pd.DataFrame([features])

        features_scaled = scaler.transform(features_df)

        prob = model.predict_proba(features_scaled)[0]

        pred = model.predict(features_scaled)[0]

        ganador = pokemon1 if pred == 1 else pokemon2

        confianza = prob[1] if pred == 1 else prob[0]

        print(f"Predicción: {ganador} gana con {confianza:.1%} de confianza")

        return ganador, confianza

    except Exception as e:

        print(f"Error al predecir: {e}")

        return None, 0



# Ejemplo

print("\nProbando el modelo SVM:")

predecir_batalla("Pikachu", "Charizard")

predecir_batalla("Blastoise", "Charizard")

# Entrenamiento final con todos los datos
model.fit(X_scaled, y) 

# Guardar modelo y escalador
joblib.dump(model, "modelo_pokemon_SVM.pkl")
joblib.dump(scaler, "scaler_pokemon_SVM.pkl")