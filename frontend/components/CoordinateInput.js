'use client'

import { AlertTriangle, CheckCircle2 } from 'lucide-react'

export default function CoordinateInput({ label, unit, value, onChange, min, max, isValid }) {
  const showState = value !== ''

  return (
    <div className="w-full">
      <div className="mb-1.5 flex items-baseline justify-between">
        <label className="text-[13px] font-medium text-slate-300">{label}</label>
        <span className="font-mono text-[11px] text-slate-500">
          {min}&deg; to {max}&deg;{unit}
        </span>
      </div>

      <div className="relative">
        <input
          type="number"
          step="0.0001"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={`e.g. ${(min + max) / 2}`}
          className={`w-full rounded-lg border bg-slate-900/60 px-3 py-2.5 pr-9 font-mono text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:ring-2 ${
            showState && !isValid
              ? 'border-red-800/60 focus:ring-red-500/40'
              : 'border-slate-700 focus:border-teal-500 focus:ring-teal-500/30'
          }`}
        />
        {showState && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2">
            {isValid ? (
              <CheckCircle2 className="h-4 w-4 text-teal-400" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-red-400" />
            )}
          </span>
        )}
      </div>

      {showState && !isValid && (
        <p className="mt-1.5 text-[11px] text-red-400">
          Must be between {min}&deg; and {max}&deg;{unit}
        </p>
      )}
    </div>
  )
}
