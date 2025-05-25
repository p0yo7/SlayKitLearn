# Run with uvicorn app:app --reload
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import openai
import os
from dotenv import load_dotenv

try:
    modelo = joblib.load("../modelos/modelo_gastos_recurrentes.pkl")
except FileNotFoundError:
    modelo = None

app = FastAPI(title="API de Gastos Recurrentes y Resumen")

load_dotenv()
openai.api_key = os.getenv("openai_key")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # o ["*"] en desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Cargar datos ===
# Si es en docker cambiar ../datos/base a datos/base...
clientes_df = pd.read_csv("datos/base_clientes_final.csv", parse_dates=["fecha_nacimiento", "fecha_alta"])
transacciones_df = pd.read_csv("datos/base_transacciones_final.csv", parse_dates=["fecha"])

# Renombrar id para evitar confusión
clientes_df.rename(columns={"id": "id_cliente"}, inplace=True)
transacciones_df.rename(columns={"id": "id_cliente"}, inplace=True)

# === Input para predicción ===
class ClienteIDInput(BaseModel):
    id_cliente: str

# === Función predictiva híbrida ===
def modelo_predictivo(cliente_id: str) -> dict:
    transacciones = transacciones_df[transacciones_df["id_cliente"] == cliente_id]

    if transacciones.empty:
        return {"mensaje": "No se encontraron transacciones para este cliente"}

    cliente_info = clientes_df[clientes_df["id_cliente"] == cliente_id]
    if cliente_info.empty:
        return {"mensaje": "No se encontró información del cliente"}

    transacciones = transacciones.sort_values("fecha")
    ultima = transacciones.iloc[-1]

    # Feature engineering
    edad = int((ultima["fecha"] - cliente_info.iloc[0]["fecha_nacimiento"]).days / 365)
    antiguedad = (ultima["fecha"] - cliente_info.iloc[0]["fecha_alta"]).days
    dias_desde_ultimo = (ultima["fecha"] - transacciones.iloc[-2]["fecha"]).days if len(transacciones) > 1 else 30
    promedio_monto = transacciones["monto"].mean()

    if modelo:
        # Crear input para el modelo
        features = pd.DataFrame([{
            "mes": ultima["fecha"].month,
            "dia_semana": ultima["fecha"].weekday(),
            "edad": edad,
            "antiguedad": antiguedad,
            "dias_desde_ultimo": dias_desde_ultimo,
            "giro_comercio": ultima["giro_comercio"],
            "tipo_venta": ultima["tipo_venta"],
            "genero": cliente_info.iloc[0]["genero"],
            "actividad_empresarial": cliente_info.iloc[0]["actividad_empresarial"],
            "tipo_persona": cliente_info.iloc[0]["tipo_persona"]
        }])

        # Hacer predicción
        dias_estimado = int(modelo["dias_model"].predict(features)[0])
        monto_estimado = round(float(modelo["monto_model"].predict(features)[0]), 2)
    else:
        # Simulación si no hay modelo
        dias_estimado = np.random.randint(5, 30)
        monto_estimado = round(np.random.normal(promedio_monto, promedio_monto * 0.2), 2)

    return {
        "cliente_id": cliente_id,
        "ultima_fecha": ultima["fecha"].date(),
        "concepto_estimado": ultima["giro_comercio"],
        "comercio_estimado": ultima["comercio"],
        "fecha_estimado": (ultima["fecha"] + timedelta(days=dias_estimado)).date(),
        "monto_estimado": monto_estimado
    }

# === Endpoint 1: Predicción de gasto recurrente ===
@app.post("/predict_gastos_recurrentes")
def predict(cliente: ClienteIDInput):
    return modelo_predictivo(cliente.id_cliente)

# === Endpoint 2: Wrapped personalizado ===
@app.get("/wrapped_gastos")
def wrapped(
    cliente_id: str,
    desde: str,
    hasta: str,
    modo: str = Query("giro_comercio", enum=["giro_comercio", "comercio"])
):
    desde = pd.to_datetime(desde)
    hasta = pd.to_datetime(hasta)

    cliente_tx = transacciones_df[
        (transacciones_df["id_cliente"] == cliente_id) &
        (transacciones_df["fecha"].between(desde, hasta))
    ]

    if cliente_tx.empty:
        return {"mensaje": "No hay transacciones en el periodo indicado"}

    resumen_gastos = (
        cliente_tx
        .groupby(modo)["monto"]
        .sum()
        .round(2)
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    resumen_categorias = (
        cliente_tx
        .groupby("tipo_venta")["monto"]
        .sum()
        .round(2)
        .to_dict()
    )

    esenciales = cliente_tx[
        cliente_tx["giro_comercio"].str.contains("SERVICIOS|SALUD|TRANSPORTE", case=False, na=False)
    ]
    esenciales_monto = esenciales["monto"].sum()
    total_monto = cliente_tx["monto"].sum()
    subs_monto = total_monto - esenciales_monto

    proporcion = [
        {"tipo": "Esenciales", "valor": round(esenciales_monto, 2)},
        {"tipo": "Suscripciones", "valor": round(subs_monto, 2)},
    ]

    predictibilidad = (
        cliente_tx
        .groupby("giro_comercio")["monto"]
        .std()
        .fillna(0)
        .apply(lambda x: 100 - min(x * 5, 100))
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
        .rename(columns={"giro_comercio": "categoria", "monto": "score"})
        .to_dict(orient="records")
    )

    compra_max = cliente_tx.loc[cliente_tx["monto"].idxmax()]
    fecha_compra = compra_max["fecha"].strftime("%-d de %B")
    monto_compra = round(compra_max["monto"], 2)
    comercio = compra_max["comercio"]

    # Llamada a OpenAI para generar texto personalizado
    prompt = (
        f"El usuario hizo su compra más memorable el {fecha_compra}, gastando ${monto_compra:,.2f} en {comercio}. "
        "Escribe una frase breve en español, amigable, tipo marketing, que destaque esta compra como un momento especial."
    )

    try:
        respuesta_ai = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un redactor creativo para una fintech joven."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.8
        )
        compra_memorable_texto = respuesta_ai["choices"][0]["message"]["content"]
    except Exception as e:
        compra_memorable_texto = f"Error al generar el mensaje: {e}"

    return {
        "cliente_id": cliente_id,
        "rango": f"{desde.date()} a {hasta.date()}",
        "moneda": "MXN",
        "total_gastado": round(total_monto, 2),
        "resumen_gastos": resumen_gastos,
        "resumen_categorias": resumen_categorias,
        "proporcion_essentials_vs_subs": proporcion,
        "predictibilidad_por_categoria": predictibilidad,
        "compra_mas_iconica": {
            "fecha": fecha_compra,
            "comercio": comercio,
            "monto": monto_compra,
            "mensaje": compra_memorable_texto
        }
    }
# === Endpoint 3: Info del Cliente ===
@app.get("/cliente_info")
def cliente_info(cliente_id: str):
    cliente = clientes_df[clientes_df["id_cliente"] == cliente_id]

    if cliente.empty:
        return {"mensaje": "Cliente no encontrado"}

    cliente = cliente.iloc[0]
    info = {
        "id_cliente": str(cliente["id_cliente"]),
        "fecha_nacimiento": cliente["fecha_nacimiento"].date(),
        "fecha_alta": cliente["fecha_alta"].date(),
        "id_municipio": int(cliente["id_municipio"]),
        "id_estado": int(cliente["id_estado"]),
        "tipo_persona": str(cliente["tipo_persona"]),
        "genero": str(cliente["genero"]),
        "actividad_empresarial": str(cliente["actividad_empresarial"])
    }

    return info

# === Endpoint 4: Resumen de transacciones ===
@app.get("/resumen_transacciones")
def resumen_transacciones(cliente_id: str, desde: str, hasta: str):
    desde = pd.to_datetime(desde)
    hasta = pd.to_datetime(hasta)

    transacciones_cliente = transacciones_df[
        (transacciones_df["id_cliente"] == cliente_id) &
        (transacciones_df["fecha"].between(desde, hasta))
    ]

    if transacciones_cliente.empty:
        return {"mensaje": "No hay transacciones en el periodo indicado"}

    resumen = { 
        "total_transacciones": len(transacciones_cliente),
        "total_gastado": round(transacciones_cliente["monto"].sum(), 2),
        "promedio_gasto": round(transacciones_cliente["monto"].mean(), 2),
        "max_gasto": round(transacciones_cliente["monto"].max(), 2),
        "min_gasto": round(transacciones_cliente["monto"].min(), 2),
        "moneda": "MXN",
        "rango": f"{desde.date()} a {hasta.date()}"
    }
    return resumen
