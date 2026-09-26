'use client'

import React from 'react'
import { useOceanStore } from '@/store/useOceanStore'
import { Thermometer, Activity, MapPin, Calendar } from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const PARAMETER_NAMES = {
  'sst': 'Sea Surface Temperature',
  'sss': 'Sea Surface Salinity',
  'ssh': 'Sea Surface Height',
  'u_current': 'Zonal Current (U)',
  'v_current': 'Meridional Current (V)',
  'u_wind': 'Zonal Wind (U)',
  'v_wind': 'Meridional Wind (V)'
}

// Helper to interpolate between two hex colors
function lerpColor(c1, c2, factor) {
  const hex2rgb = (hex) => {
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return [r, g, b]
  }
  const rgb1 = hex2rgb(c1)
  const rgb2 = hex2rgb(c2)
  const r = Math.round(rgb1[0] + factor * (rgb2[0] - rgb1[0]))
  const g = Math.round(rgb1[1] + factor * (rgb2[1] - rgb1[1]))
  const b = Math.round(rgb1[2] + factor * (rgb2[2] - rgb1[2]))
  return `rgb(${r}, ${g}, ${b})`
}

// Dynamically map temperature where Surface = Orange and Bottom = Blue
function getDynamicTemperatureColor(temp, surfaceTemp, bottomTemp) {
  if (temp === null || temp === undefined || surfaceTemp === null || bottomTemp === null) return 'rgb(50,50,50)'
  if (surfaceTemp === bottomTemp) return '#ff9800'
  
  let factor = (temp - bottomTemp) / (surfaceTemp - bottomTemp)
  factor = Math.max(0, Math.min(1, factor)) // Clamp between 0 and 1
  
  // 3-point lerp to avoid ugly muddy colors in the middle (Blue -> Cyan -> Orange)
  if (factor <= 0.5) {
    return lerpColor('#005f89', '#00bfa5', factor * 2) // Deep Blue to Cyan
  } else {
    return lerpColor('#00bfa5', '#ff9800', (factor - 0.5) * 2) // Cyan to Orange
  }
}

export default function ProfileVisualizer() {
  const { predictionData, isPredicting, predictionError } = useOceanStore()

  if (isPredicting) {
    return (
      <div className="mt-8 flex h-64 flex-col items-center justify-center rounded-2xl border border-slate-800 bg-slate-900/40 p-8 shadow-2xl">
        <Activity className="h-8 w-8 animate-pulse text-teal-400 mb-4" />
        <p className="text-slate-300 font-medium animate-pulse">Running Neural Network Inference...</p>
      </div>
    )
  }

  if (predictionError) {
    return (
      <div className="mt-8 rounded-2xl border border-red-900/50 bg-red-950/20 p-6 text-center shadow-2xl">
        <p className="text-red-400 font-medium">Error: {predictionError}</p>
      </div>
    )
  }

  if (!predictionData) return null

  const profile = predictionData.profile
  const meta = predictionData.metadata
  const maxDepth = 1000 // Fixed as per our model (15 levels up to 1000m)

  // Find the anchors for the dynamic gradient
  const surfaceTemp = profile.find(p => p.depth_m === 0)?.temperature_c ?? profile[0].temperature_c
  const bottomTemp = profile.find(p => p.depth_m === maxDepth)?.temperature_c ?? profile[profile.length - 1].temperature_c

  // Build the CSS linear-gradient string interpolating through all known points
  const gradientStops = profile
    .filter(p => p.temperature_c !== null)
    .map(p => {
      const percentage = (p.depth_m / maxDepth) * 100
      const color = getDynamicTemperatureColor(p.temperature_c, surfaceTemp, bottomTemp)
      return `${color} ${percentage}%`
    })
    .join(', ')

  return (
    <div className="mt-8 overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl">
      {/* Header */}
      <div className="border-b border-slate-800 bg-slate-900/50 p-4 sm:px-6">
        <h2 className="flex items-center gap-2 font-display text-lg font-semibold text-slate-100">
          <Thermometer className="h-5 w-5 text-teal-500" />
          Subsurface Temperature Profile
        </h2>
        
        <div className="mt-3 flex flex-wrap gap-4 text-sm text-slate-400">
          <div className="flex items-center gap-1.5 rounded-full bg-slate-800/50 px-3 py-1">
            <MapPin className="h-3.5 w-3.5 text-slate-500" />
            Lat: {meta.actual_grid_lat.toFixed(2)}°, Lon: {meta.actual_grid_lon.toFixed(2)}°
          </div>
          <div className="flex items-center gap-1.5 rounded-full bg-slate-800/50 px-3 py-1">
            <Calendar className="h-3.5 w-3.5 text-slate-500" />
            {meta.date}
          </div>
        </div>
      </div>

      {/* Main Content: 3-Column Layout */}
      <div className="p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-5 items-start">
        
        {/* Left: Input Parameters (Span 4) */}
        <div className="hidden lg:flex flex-col h-[400px] rounded-xl border border-slate-700/50 bg-slate-900/40 shadow-inner overflow-hidden lg:col-span-4">
          <div className="bg-slate-800/80 px-4 py-2.5 border-b border-slate-700/50">
            <h3 className="text-sm font-semibold tracking-wider text-slate-300 uppercase text-center">
              Satellite Inputs
            </h3>
          </div>
          
          <div className="p-5 flex-1 flex flex-col overflow-hidden">
            {predictionData.inputs ? (
            <div className="grid grid-cols-1 gap-2 overflow-y-auto custom-scrollbar pr-2">
              {Object.entries(predictionData.inputs).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between bg-slate-950/50 rounded-lg p-2.5 border border-slate-800/80">
                  <span className="text-slate-400 text-[11px] uppercase tracking-wide">
                    {PARAMETER_NAMES[key] || key.replace('_', ' ')}
                  </span>
                  <span className="text-slate-200 font-semibold text-xs">
                    {value !== null ? value.toFixed(3) : '--'}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <p className="text-slate-600 text-xs text-center italic">No input data available</p>
            </div>
          )}
          </div>
        </div>

        {/* Middle: Tabular readouts, now narrower to reduce empty space (Span 5) */}
        <div className="w-full flex flex-col h-[400px] bg-slate-950/30 rounded-xl shadow-inner border border-slate-800/60 overflow-hidden lg:col-span-5">
          <div className="bg-slate-800/80 px-4 py-2.5 border-b border-slate-700/50">
            <h3 className="text-sm font-semibold tracking-wider text-slate-300 uppercase text-center">
              Prediction
            </h3>
          </div>
          <div className="p-1 flex-1 flex flex-col overflow-hidden">
            <div className="grid grid-cols-2 gap-px bg-slate-800/60 font-medium text-xs text-slate-400 uppercase tracking-wider">
            <div className="bg-slate-900 py-2 px-4">Depth (m)</div>
            <div className="bg-slate-900 py-2 px-4 text-right">Temp (°C)</div>
          </div>
          
          <div className="max-h-[360px] overflow-y-auto custom-scrollbar">
            {profile.map((p, idx) => (
              <div 
                key={idx} 
                className="grid grid-cols-2 gap-px bg-slate-800/20 text-sm hover:bg-slate-800/40 transition-colors"
              >
                <div className="bg-slate-900/80 py-2.5 px-4 text-slate-300 font-medium">
                  {p.depth_m.toFixed(1)}
                </div>
                <div className="bg-slate-900/80 py-2.5 px-4 text-right text-slate-100 flex items-center justify-end gap-2">
                  <div 
                    className="w-2.5 h-2.5 rounded-full shadow-sm"
                    style={{ backgroundColor: getDynamicTemperatureColor(p.temperature_c, surfaceTemp, bottomTemp) }}
                  />
                  {p.temperature_c !== null ? p.temperature_c.toFixed(2) + '°' : '--'}
                </div>
              </div>
            ))}
            </div>
          </div>
        </div>

        {/* Right: Widened gradient color bar (Span 3) */}
        <div className="relative h-[400px] w-full flex justify-end gap-4 lg:col-span-3">
          
          {/* Depth axis ticks */}
          <div className="relative h-full flex flex-col justify-between text-xs font-medium text-slate-500 pb-2 text-right">
            <span>0m</span>
            <span>250m</span>
            <span>500m</span>
            <span>750m</span>
            <span>1000m</span>
          </div>

          {/* Widened gradient meter */}
          <div 
            className="h-full w-20 sm:w-28 rounded-2xl shadow-[inset_0_2px_15px_rgba(0,0,0,0.6)] border border-slate-700/50"
            style={{ background: `linear-gradient(to bottom, ${gradientStops})` }}
          />
          
        </div>

      </div>

      {/* Chart Section */}
      <div className="border-t border-slate-800 bg-slate-900/40 p-4 sm:p-6 sm:px-8">
        <h3 className="text-sm font-semibold tracking-wider text-slate-400 uppercase mb-6 border-b border-slate-800 pb-2">
          Depth vs Temperature Curve
        </h3>
        <div className="h-[280px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={profile} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <defs>
                <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00bfa5" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#00bfa5" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis 
                dataKey="depth_m" 
                stroke="#64748b" 
                tick={{fill: '#64748b', fontSize: 11}} 
                tickMargin={10}
                label={{ value: 'Depth (m)', position: 'insideBottom', offset: -15, fill: '#94a3b8', fontSize: 13, fontWeight: 500 }} 
              />
              <YAxis 
                stroke="#64748b" 
                tick={{fill: '#64748b', fontSize: 11}}
                domain={['dataMin - 1', 'dataMax + 1']}
                tickFormatter={(val) => val.toFixed(1)}
                label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft', offset: 15, fill: '#94a3b8', fontSize: 13, fontWeight: 500 }}
              />
              <Tooltip 
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', color: '#f8fafc', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' }}
                itemStyle={{ color: '#00bfa5', fontWeight: 600 }}
                labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
                labelFormatter={(label) => `Depth: ${label}m`}
                formatter={(value) => [`${Number(value).toFixed(2)} °C`, 'Temperature']}
              />
              <Area 
                type="monotone" 
                dataKey="temperature_c" 
                stroke="#00bfa5" 
                strokeWidth={3} 
                fillOpacity={1} 
                fill="url(#colorTemp)" 
                activeDot={{ r: 6, fill: '#00bfa5', stroke: '#0f172a', strokeWidth: 3 }} 
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
