'use client'

import { useState } from 'react'
import { CalendarDays, Send, MapPin, Crosshair } from 'lucide-react'
import { useOceanStore, isLatValid, isLonValid } from '@/store/useOceanStore'

export default function FloatingPanel() {
  const date = useOceanStore((state) => state.date)
  const latitude = useOceanStore((state) => state.latitude)
  const longitude = useOceanStore((state) => state.longitude)
  const setDate = useOceanStore((state) => state.setDate)
  const isReadyToSubmit = useOceanStore((state) => state.isReadyToSubmit)
  const submitPayload = useOceanStore((state) => state.submitPayload)

  const [justSubmitted, setJustSubmitted] = useState(false)

  const hasPin = isLatValid(latitude) && isLonValid(longitude)
  const ready = isReadyToSubmit()

  const handleSubmit = () => {
    if (!ready) return
    submitPayload()
    setJustSubmitted(true)
    setTimeout(() => setJustSubmitted(false), 2500)
  }

  return (
    <div className="flex w-full flex-col rounded-2xl border border-slate-800 bg-slate-950/90 shadow-2xl backdrop-blur">
      <div className="border-b border-slate-800 px-5 py-4">
        <h2 className="font-display text-base font-semibold text-slate-100">Request settings</h2>
        <p className="mt-0.5 text-xs text-slate-500">Choose an observation date and pin a location</p>
      </div>

      <div className="px-5 py-4">
        <div className="mb-5 flex items-center gap-3 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-3">
          {hasPin ? (
            <MapPin className="h-4 w-4 shrink-0 text-teal-400" />
          ) : (
            <Crosshair className="h-4 w-4 shrink-0 animate-pulse text-slate-500" />
          )}
          <div className="min-w-0 font-mono text-xs">
            {hasPin ? (
              <span className="text-slate-200">
                {Number(latitude).toFixed(4)}&deg;N, {Number(longitude).toFixed(4)}&deg;E
              </span>
            ) : (
              <span className="text-slate-500">Click the ocean to set a location</span>
            )}
          </div>
        </div>

        <div className="mb-5">
          <label className="mb-1.5 block text-[13px] font-medium text-slate-300">
            Observation date
          </label>
          <div className="relative">
            <CalendarDays className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <input
              type="date"
              value={date}
              onChange={(event) => setDate(event.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-900/60 py-2.5 pl-9 pr-3 text-sm text-slate-100 focus:border-teal-500 focus:outline-none focus:ring-2 focus:ring-teal-500/30 [color-scheme:dark]"
            />
          </div>
        </div>
      </div>

      <div className="border-t border-slate-800 px-5 py-4">
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!ready}
          className={`flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold shadow-glow transition-all ${
            ready
              ? 'bg-gradient-to-br from-teal-600 to-cyan-700 text-white hover:from-teal-500 hover:to-cyan-600'
              : 'cursor-not-allowed bg-slate-800 text-slate-500 shadow-none'
          }`}
        >
          <Send className="h-4 w-4" />
          {justSubmitted ? 'Request prepared' : 'Get result'}
        </button>
        {(!hasPin || !date) && (
          <p className="mt-2 text-center text-[11px] text-slate-500">
            {!hasPin ? 'Pin a location' : 'Choose a date'} to continue
          </p>
        )}
      </div>
    </div>
  )
}
