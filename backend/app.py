from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

try:
    modelo = joblib.load("../modelos/modelo_gastos_recurrentes.pkl")
except FileNotFoundError:
    modelo = None

app = FastAPI(title="API de Gastos Recurrentes y Resumen")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # o ["*"] en desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Cargar datos ===
clientes_df = pd.read_csv("../datos/base_clientes_final.csv", parse_dates=["fecha_nacimiento", "fecha_alta"])
transacciones_df = pd.read_csv("../datos/base_transacciones_final.csv", parse_dates=["fecha"])

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

    resumen_raw = (
        cliente_tx
        .groupby(modo)["monto"]
        .sum()
        .round(2)
        .sort_values(ascending=False)
    )

    top_5_resumen = resumen_raw.head(5).to_dict()
    total_gastado = round(cliente_tx["monto"].sum(), 2)

    return {
        "cliente_id": cliente_id,
        "rango": f"{desde.date()} a {hasta.date()}",
        "moneda": "MXN",
        "total_gastado": total_gastado,
        "resumen_gastos": top_5_resumen
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
