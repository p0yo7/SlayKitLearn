'use client'

import { useEffect, useState } from 'react'

interface WrappedResponse {
  cliente_id: string
  rango: string
  moneda: string
  total_gastado: number
  resumen_gastos: Record<string, number>
  resumen_categorias: Record<string, number>
  proporcion_essentials_vs_subs: { tipo: string; valor: number }[]
  predictibilidad_por_categoria: { categoria: string; score: number }[]
  compra_mas_iconica?: {
    fecha: string
    comercio: string
    monto: number
    mensaje: string
  }
}

export default function WrappedSummary() {
  const [data, setData] = useState<WrappedResponse | null>(null)

  useEffect(() => {
    fetch('http://localhost:8000/wrapped_gastos?cliente_id=9980f12e32711330d5f58460e169e6207afda041&desde=2020-01-01&hasta=2024-12-31')
      .then(res => res.json())
      .then(setData)
  }, [])

  if (!data) return <p className="text-gray-500">Cargando...</p>

  const topNegocios = Object.entries(data.resumen_gastos)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3)

  const totalCategoria = Object.values(data.resumen_categorias).reduce((a, b) => a + b, 0)
  const canalDigital = data.resumen_categorias['digital'] || 0
  const canalFisico = data.resumen_categorias['físico'] || 0
  const canalDigitalPct = Math.round((canalDigital / (canalDigital + canalFisico)) * 100)
  const canalFisicoPct = 100 - canalDigitalPct

  const predictibilidadPromedio = Math.round(
    data.predictibilidad_por_categoria.reduce((a, b) => a + b.score, 0) / data.predictibilidad_por_categoria.length
  )

  return (
    <div className="bg-pink-700 text-white min-h-screen p-10 font-sans">
      <h1 className="text-4xl font-bold mb-8">hey, <span className="text-white">Wrap</span></h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
        <div>
          <h2 className="text-xl font-semibold">Top Negocios</h2>
          <ul className="mt-2">
            {topNegocios.map(([name], i) => (
              <li key={i}>{i + 1}. <span className="font-bold">{name}</span></li>
            ))}
          </ul>

          <div className="mt-8">
            <h2 className="text-xl font-semibold">Canal Favorito</h2>
            <p className="mt-1">Digital: {canalDigitalPct}%</p>
            <p>Físico: {canalFisicoPct}%</p>
            <p className="mt-2">Prefieres la eficiencia de las plataformas digitales.</p>
          </div>

          <div className="mt-8">
            <h2 className="text-xl font-semibold">
              ¿Qué tan predecible eres?
            </h2>
            <p className="mt-2">
              Tu gasto es {predictibilidadPromedio}% constante, con algunos picos de consumo marcados por eventos.
            </p>
          </div>
        </div>

        <div>
          <div>
            <h2 className="text-xl font-semibold">Tus prioridades</h2>
            <p className="mt-2">Tu dinero habla:</p>
            {data.proporcion_essentials_vs_subs.map((item, i) => (
              <p key={i}>- {item.valor}% en {item.tipo.toLowerCase()}</p>
            ))}
          </div>

          <div className="mt-12">
            <h2 className="text-xl font-semibold">Gasto más icónico</h2>
            <p className="mt-2">
              {data.compra_mas_iconica?.mensaje ?? 'No se encontró información sobre la compra más memorable.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
