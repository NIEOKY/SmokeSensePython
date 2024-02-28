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
        return "Mi prediccion es que eres una persona que no fuma, ¡felicidades!👍"
    elif prediccion == 2:
        print("La persona es fumadora casual")
        return "Mi prediccion es que eres una persona que fuma casualmente, ¡cuidado con tu salud!🚬"
    elif prediccion == 1:
        print("La persona fuma muy seguido")
        return "Mi prediccion es que eres una persona que fuma muy seguido, cuidado con tu salud!💀"
