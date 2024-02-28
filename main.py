from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import data_preprocessing as dp
import model_training as mt
import predict
from fastapi.middleware.cors import CORSMiddleware
# Instanciar la aplicación FastAPI en puerto 5000
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://smoke-sense.vercel.app",
    "https://smoke-sense-git-master-nieoky.vercel.app/",
    "https://smoke-sense-2n97cohtv-nieoky.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
    return {resultado}

# Opcional: Endpoint para verificar el estado de la API
@app.get("/")
async def read_root():
    return {"mensaje": "API de predicción de hábitos de fumado funcionando"}


#uvicorn main:app --reload
#ngrok http --domain=greatly-apt-bluejay.ngrok-free.app 8000