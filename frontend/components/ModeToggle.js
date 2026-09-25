'use client'

import { ClipboardList, MapPinned } from 'lucide-react'
import { useOceanStore } from '@/store/useOceanStore'

const OPTIONS = [
  { id: 'manual', label: 'Manual Input', icon: ClipboardList },
  { id: 'map', label: 'Interactive Map', icon: MapPinned }
]

export default function ModeToggle() {
  const mode = useOceanStore((state) => state.mode)
  const setMode = useOceanStore((state) => state.setMode)

  return (
    <div className="inline-flex rounded-xl border border-slate-800 bg-slate-900/60 p-1">
      {OPTIONS.map((option) => {
        const Icon = option.icon
        const active = mode === option.id
        return (
          <button
            key={option.id}
            type="button"
            onClick={() => setMode(option.id)}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
              active
                ? 'bg-gradient-to-br from-teal-600 to-cyan-700 text-white shadow-glow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Icon className="h-4 w-4" />
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
