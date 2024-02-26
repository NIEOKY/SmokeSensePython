import pandas as pd

def load_and_clean_data(filepath):
    # Cargar datos
    df = pd.read_csv(filepath)

    # Seleccionar columnas relevantes
    columnas_seleccionadas = [
        'ds3', 'ds2', 'ds6', 'ds7', 'ds8', 'ds9', 'ds10', 'ds16', 'ds21', 'tb02', 'al1',
    ]
    df_limpio = df[columnas_seleccionadas]

    # Renombrar columnas
    df_limpio.columns = [
        'edad', 'sexo', 'estado_civil', 'religion', 'es_estudiante', 'ultimo_grado', 
        'trabaja', 'ocupacion', 'tiene_hijos', 'frecuencia_fuma', 'alcohol',
    ]

    # Limpieza de datos
    df_limpio['ultimo_grado'].fillna(df_limpio['ultimo_grado'].mean(), inplace=True)
    df_filtrado = df_limpio[(df_limpio['frecuencia_fuma'] != 7) & (df_limpio['frecuencia_fuma'] != 9)]
    df_limpio = df_filtrado

    return df_limpio

def balance_data(df_limpio, objetivo_muestras=15000):
    df_clase_1 = df_limpio[df_limpio['frecuencia_fuma'] == 1]
    df_clase_2 = df_limpio[df_limpio['frecuencia_fuma'] == 2]
    df_clase_3 = df_limpio[df_limpio['frecuencia_fuma'] == 3]

    df_clase_1_sobremuestreada = df_clase_1.sample(objetivo_muestras, replace=True, random_state=42)
    df_clase_2_sobremuestreada = df_clase_2.sample(objetivo_muestras, replace=True, random_state=42)

    if len(df_clase_3) >= objetivo_muestras:
        df_clase_3_muestra = df_clase_3.sample(objetivo_muestras, random_state=42)
    else:
        df_clase_3_muestra = df_clase_3.sample(objetivo_muestras, replace=True, random_state=42)

    df_balanceado = pd.concat([df_clase_1_sobremuestreada, df_clase_2_sobremuestreada, df_clase_3_muestra])
    df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanceado
