from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import data_preprocessing as dp
import model_training as mt
import predict
from sklearn.preprocessing import StandardScaler

# Instanciar la aplicación FastAPI
app = FastAPI()

# Definir el modelo de Pydantic para los datos de entrada esperados
class DatosEntrada(BaseModel):
    edad: int
    sexo: int
    estado_civil: int
    religion: int
    es_estudiante: int
    ultimo_grado: int
    trabaja: int
    ocupacion: int
    tiene_hijos: int
    alcohol: int

# Cargar y preprocesar datos, entrenar modelo (Esto podría moverse a un endpoint de inicialización o ejecutarse fuera de la API)
df = dp.load_and_clean_data("datos.csv")
df_balanceado = dp.balance_data(df)
modelo_rf,scaler = mt.train_model(df_balanceado)

@app.post("/predict/")
async def hacer_prediccion(datos: DatosEntrada):
    entrada = datos.dict().values()
    if len(entrada) != 10:
        raise HTTPException(status_code=400, detail="Número incorrecto de características en la entrada.")
    prediccion = predict.hacer_prediccion(modelo_rf, scaler, list(entrada))
    resultado = predict.mostrar_prediccion(prediccion)
    return {"prediccion": resultado}

# Opcional: Endpoint para verificar el estado de la API
@app.get("/")
async def read_root():
    return {"mensaje": "API de predicción de hábitos de fumado funcionando"}
