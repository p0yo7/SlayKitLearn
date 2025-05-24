from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = FastAPI(title="API de Gastos Recurrentes y Resumen")

# === Cargar datos ===
clientes_df = pd.read_csv("../datos/base_clientes_final.csv", parse_dates=["fecha_nacimiento", "fecha_alta"])
transacciones_df = pd.read_csv("../datos/base_transacciones_final.csv", parse_dates=["fecha"])

# Renombrar id para evitar confusión
clientes_df.rename(columns={"id": "id_cliente"}, inplace=True)
transacciones_df.rename(columns={"id": "id_cliente"}, inplace=True)

# === Input para predicción ===
class ClienteIDInput(BaseModel):
    id_cliente: str

# === Simulación de modelo predictivo ===
def modelo_predictivo(cliente_id: str) -> dict:
    transacciones = transacciones_df[transacciones_df["id_cliente"] == cliente_id]

    if transacciones.empty:
        return {"mensaje": "No se encontraron transacciones para este cliente"}

    transacciones = transacciones.sort_values("fecha")

    ultima = transacciones.iloc[-1]
    promedio_monto = transacciones["monto"].mean()
    dias = np.random.randint(5, 30)
    proxima_fecha = ultima["fecha"] + timedelta(days=dias)

    return {
        "cliente_id": cliente_id,
        "ultima_fecha": ultima["fecha"].date(),
        "concepto_estimado": ultima["giro_comercio"],
        "comercio_estimado": ultima["comercio"],
        "fecha_estimado": proxima_fecha.date(),
        "monto_estimado": round(np.random.normal(promedio_monto, promedio_monto * 0.2), 2)
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

    resumen_raw = cliente_tx.groupby(modo)["monto"].sum()
    resumen = resumen_raw.round(2).sort_values(ascending=False).to_dict()

    total_gastado = round(cliente_tx["monto"].sum(), 2)

    return {
        "cliente_id": cliente_id,
        "rango": f"{desde.date()} a {hasta.date()}",
        "moneda": "MXN",
        "total_gastado": total_gastado,
        "resumen_gastos": resumen
    }
