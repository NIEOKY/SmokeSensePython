import pandas as pd

def hacer_prediccion(rf, scaler, entrada):
    nombres_columnas = ['edad', 'sexo', 'estado_civil', 'religion', 'es_estudiante', 'ultimo_grado', 'trabaja', 'ocupacion', 'tiene_hijos', 'alcohol']
    entrada_df = pd.DataFrame([entrada], columns=nombres_columnas)
    entrada_escalada = scaler.transform(entrada_df)
    prediccion = rf.predict(entrada_escalada)
    return prediccion

def mostrar_prediccion(prediccion):
    if prediccion == 3:
        print("La persona no fuma")
        return "La persona no fuma"
    elif prediccion == 2:
        print("La persona es fumadora casual")
        return "La persona es fumadora casual"
    elif prediccion == 1:
        print("La persona fuma muy seguido")
        return "La persona fuma muy seguido"
