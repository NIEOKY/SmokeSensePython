import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

ruta_archivo = 'datos.csv'
df = pd.read_csv(ruta_archivo)

print(df.head())

columnas_seleccionadas = [
    'ds3',  # Edad
    'ds2',  # Sexo
    'ds6',  # Estado civil
    'ds7',  # Religión
    'ds8',  # Es estudiante
    'ds9',  # Último grado completado
    'ds10', # Trabaja
    'ds16', # Ocupación
    'ds21', # Tiene hijos
    'tb02', # Frecuencia fuma (Variable independiente)
    'al1',  # Bebida alcohólica
]

# Filtrar el DataFrame para mantener solo las columnas seleccionadas
df_limpio = df[columnas_seleccionadas]

# Renombrar las columnas del DataFrame
df_limpio.columns = [
    'edad',  # ds3
    'sexo',  # ds2
    'estado_civil',  # ds6
    'religion',  # ds7
    'es_estudiante',  # ds8
    'ultimo_grado',  # ds9
    'trabaja',  # ds10
    'ocupacion',  # ds16
    'tiene_hijos',  # ds21
    'frecuencia_fuma',  # tb02
    'alcohol',  # al1
]

# Mostrar las primeras filas del DataFrame limpio para verificar
print(df_limpio.head())

nan_por_columna = df_limpio.isna().sum()

print(nan_por_columna)

if nan_por_columna.sum() > 0:
    df_limpio['ultimo_grado'].fillna(df_limpio['ultimo_grado'].mean(), inplace=True)
    df_filtrado = df_limpio[(df_limpio['frecuencia_fuma'] != 7) & (df_limpio['frecuencia_fuma'] != 9)]
    df_limpio = df_filtrado
    nan_por_columna = df_limpio.isna().sum()
    print(nan_por_columna)
conteos = df_limpio['frecuencia_fuma'].value_counts()
print(conteos)


import pandas as pd

# Calcula el número de muestras que quieres para cada clase
objetivo_muestras = 15000  #si quieres tener aproximadamente 15000 muestras para cada clase

# Filtra el DataFrame por clases
df_clase_1 = df_limpio[df_limpio['frecuencia_fuma'] == 1]
df_clase_2 = df_limpio[df_limpio['frecuencia_fuma'] == 2]
df_clase_3 = df_limpio[df_limpio['frecuencia_fuma'] == 3]

# Sobremuestrea las clases 1 y 2 replicando algunas filas al azar hasta alcanzar el número objetivo de muestras
df_clase_1_sobremuestreada = df_clase_1.sample(objetivo_muestras, replace=True, random_state=42)
df_clase_2_sobremuestreada = df_clase_2.sample(objetivo_muestras, replace=True, random_state=42)

# Para la clase 3, toma una muestra de tamaño 'objetivo_muestras' si es posible, de lo contrario usa replace=True
if len(df_clase_3) >= objetivo_muestras:
    df_clase_3_muestra = df_clase_3.sample(objetivo_muestras, random_state=42)
else:
    df_clase_3_muestra = df_clase_3.sample(objetivo_muestras, replace=True, random_state=42)

# Combina las clases sobremuestreadas con la muestra de la clase 3
df_balanceado = pd.concat([df_clase_1_sobremuestreada, df_clase_2_sobremuestreada, df_clase_3_muestra])

# Mezcla el DataFrame resultante para evitar cualquier orden de clase
df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

# Verifica los nuevos conteos de clases
conteos_balanceados = df_balanceado['frecuencia_fuma'].value_counts()
print(conteos_balanceados)

df_limpio = df_balanceado

from sklearn.model_selection import train_test_split

X = df_limpio.drop('frecuencia_fuma', axis=1)  # Variables predictoras
y = df_limpio['frecuencia_fuma']  # Variable a predecir

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#random forest
param_grid = {
    'n_estimators': 200,  # Number of trees in the forest
    'max_depth': None,  # Maximum depth of the tree
    'min_samples_split': 2,  # Minimum number of samples required to split an internal node
    'min_samples_leaf': 1,  # Minimum number of samples required to be at a leaf node
    'bootstrap': False  # Whether bootstrap samples are used when building trees
}

rf = RandomForestClassifier(**param_grid)

rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print('Accuracy Random Forest:', accuracy_rf)

valores_de_entrada = [[
    22,  # Edad
    1,   # Sexo (1 para masculino, 2 para femenino)
    6,   # Estado Civil(1Casado(a) 2Unión Libre 3Separado(a) 4Divorciado(a) 5viudo(a) 6Soltero(a))
    1,   # Religión 1Católica 2Protestante o Evangélica 3Judaica 4Cristiana 5Otra   6Ninguna religión
    3,   # 1"No, nunca ha asistido a la escuela" 2"No, pero si fue a la escuela" 3SÍ
    6,   # Último Grado Completado
            #(1Primaria incompleta (1 a 5 años)
            #2Primaria completa (6 años)
            #3Secundaria incompleta (1 a 2 años)
            #4Secundaria completa o equivalente  (3 años)
            #5Bachillerato incompleto (1 a 2 años)
            #6Bachillerato completo o equivalente (aprox. 3 años)
            #7Estudios Universitarios incompletos (1 a 3 años)
            #8Estudios  Universitarios completos (4 a 5 años)
            #9Estudios de Posgrado (2 a 4 años)
            #10No contesta)
    1,   # Trabaja (asumiendo 1 para sí, 2 para no)
    1,   # Ocupación
            #1. Profesionista (con estudios universitarios, maestro universit),
            #2. Maestro (de primaria, secundaria, preparatoria, etc.),
            #3. Director o propietario de empresa o negocio,
            #4. Propietario de pequeño comercio (tienda, restaurante, miscel),
            #5. Empleado de banco, oficina, establecimiento o dependencias g,
            #6. Obrero calificado (tornero, mecánico encuadernador, etc),
            #7. Obrero no calificado, con trabajo eventual, cabo, soldado ra,
            #8. Agricultor,
            #9. Campesino,
            #10. Subempleado (vendedor no asalariado, bolero, lavacoches, jor,
            #11. Estudiante,
            #12. Ama de casa,
            #13. Pensionado o jubilado,
            #14. Incapacidad permanente,
            #15. Otro)
    1,   # Tiene Hijos (asumiendo 1 para sí, 2 para no)
    1,   # Bebida Alcohólica (1 ha tomado 2 no)
]]

nombres_columnas = ['edad', 'sexo', 'estado_civil', 'religion', 'es_estudiante', 'ultimo_grado', 'trabaja', 'ocupacion', 'tiene_hijos', 'alcohol']
entrada_df = pd.DataFrame(valores_de_entrada, columns=nombres_columnas)

entrada_escalada = scaler.transform(entrada_df)

prediccion = rf.predict(entrada_escalada)

# Imprimir la predicción
print(f"La predicción para la nueva muestra es: {prediccion}")
if(prediccion == 3):
    print("la persona no fuma")

if(prediccion ==2):
    print("la persona es fumadora casual")

if(prediccion == 1):
    print("la persona fuma muy seguido")
# Si el modelo soporta predict_proba y deseas ver las probabilidades para cada clase, puedes hacer lo siguiente:
if hasattr(rf, "predict_proba"):
    probabilidades = rf.predict_proba(entrada_escalada)
    print("Probabilidades para cada clase:", probabilidades)
