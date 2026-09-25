'use client'

import { useState } from 'react'
import { CalendarDays, Loader2, Send, RotateCcw } from 'lucide-react'
import { useOceanStore, isLatValid, isLonValid } from '@/store/useOceanStore'
import { FILE_FIELDS, NIO_BOUNDS } from '@/lib/constants'
import CoordinateInput from './CoordinateInput'
import FileUploadField from './FileUploadField'

export default function ManualInputForm() {
  const date = useOceanStore((state) => state.date)
  const latitude = useOceanStore((state) => state.latitude)
  const longitude = useOceanStore((state) => state.longitude)
  const files = useOceanStore((state) => state.files)
  const setDate = useOceanStore((state) => state.setDate)
  const setLatitude = useOceanStore((state) => state.setLatitude)
  const setLongitude = useOceanStore((state) => state.setLongitude)
  const setFile = useOceanStore((state) => state.setFile)
  const removeFile = useOceanStore((state) => state.removeFile)
  const resetForm = useOceanStore((state) => state.resetForm)
  const isReadyToSubmit = useOceanStore((state) => state.isReadyToSubmit)
  const submitPayload = useOceanStore((state) => state.submitPayload)

  const [isSubmitting, setIsSubmitting] = useState(false)
  const [justSubmitted, setJustSubmitted] = useState(false)

  const filesUploaded = Object.values(files).filter(Boolean).length
  const ready = isReadyToSubmit()

  const handleSubmit = async () => {
    if (!ready) return
    setIsSubmitting(true)
    submitPayload()
    await new Promise((resolve) => setTimeout(resolve, 600))
    setIsSubmitting(false)
    setJustSubmitted(true)
    setTimeout(() => setJustSubmitted(false), 2500)
  }

  return (
    <div className="mx-auto max-w-4xl">
      <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 sm:p-8">
        <section className="mb-8">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
            Observation details
          </h2>

          <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
            <div>
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

            <CoordinateInput
              label="Latitude"
              unit="N"
              value={latitude}
              onChange={setLatitude}
              min={NIO_BOUNDS.latMin}
              max={NIO_BOUNDS.latMax}
              isValid={isLatValid(latitude)}
            />

            <CoordinateInput
              label="Longitude"
              unit="E"
              value={longitude}
              onChange={setLongitude}
              min={NIO_BOUNDS.lonMin}
              max={NIO_BOUNDS.lonMax}
              isValid={isLonValid(longitude)}
            />
          </div>
        </section>

        <section className="mb-8">
          <div className="mb-4 flex items-baseline justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              NetCDF data layers
            </h2>
            <span className="font-mono text-xs text-slate-500">{filesUploaded} / 7 uploaded</span>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {FILE_FIELDS.map((field) => (
              <FileUploadField
                key={field.key}
                field={field}
                file={files[field.key]}
                onSelect={setFile}
                onRemove={removeFile}
              />
            ))}
          </div>
        </section>

        <div className="flex flex-col-reverse items-stretch gap-3 border-t border-slate-800 pt-6 sm:flex-row sm:items-center sm:justify-between">
          <button
            type="button"
            onClick={resetForm}
            className="flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
          >
            <RotateCcw className="h-4 w-4" />
            Clear form
          </button>

          <button
            type="button"
            onClick={handleSubmit}
            disabled={!ready || isSubmitting}
            className={`flex items-center justify-center gap-2 rounded-lg px-6 py-2.5 text-sm font-semibold shadow-glow transition-all ${
              ready && !isSubmitting
                ? 'bg-gradient-to-br from-teal-600 to-cyan-700 text-white hover:from-teal-500 hover:to-cyan-600'
                : 'cursor-not-allowed bg-slate-800 text-slate-500 shadow-none'
            }`}
          >
            {isSubmitting ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            {justSubmitted ? 'Payload logged to console' : 'Process Ocean Data'}
          </button>
        </div>
      </div>
    </div>
  )
}
