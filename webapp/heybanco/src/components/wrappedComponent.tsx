'use client'

import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

type WrappedResponse = {
  cliente_id: string
  rango: string
  moneda: string
  total_gastado: number
  resumen_gastos: { [comercio: string]: number }
}

export default function WrappedChart() {
  const [data, setData] = useState<WrappedResponse | null>(null)

  useEffect(() => {
    fetch('http://localhost:8000/wrapped_gastos?cliente_id=9980f12e32711330d5f58460e169e6207afda041&desde=2020-01-01&hasta=2024-12-31&modo=comercio')
      .then(res => res.json())
      .then(setData)
  }, [])
  
  if (!data) return <p className="text-gray-500">Cargando...</p>

  // Convertimos el objeto resumen_gastos a un array para graficarlo
  const chartData = Object.entries(data.resumen_gastos).map(([comercio, monto]) => ({
    comercio,
    monto: parseFloat(monto.toFixed(2))
  }))

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-2">Gastos por Comercio ({data.rango})</h2>
      <p className="text-sm mb-4 text-gray-700">Total gastado: {data.total_gastado.toLocaleString()} {data.moneda}</p>

      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 20, right: 20, left: 60, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="comercio" width={140} />
          <Tooltip />
          <Legend />
          <Bar dataKey="monto" fill="#8884d8" name="Monto gastado" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
