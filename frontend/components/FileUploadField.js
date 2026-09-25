'use client'

import { useState, useCallback } from 'react'
import { UploadCloud, FileCheck2, X, AlertTriangle } from 'lucide-react'

const formatSize = (bytes) => {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function FileUploadField({ field, file, onSelect, onRemove }) {
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState('')
  const Icon = field.icon

  const validateAndSet = useCallback(
    (selected) => {
      if (!selected) return
      if (!selected.name.toLowerCase().endsWith('.nc')) {
        setError('Only .nc (NetCDF) files are accepted')
        return
      }
      setError('')
      onSelect(field.key, selected)
    },
    [field.key, onSelect]
  )

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    validateAndSet(event.dataTransfer.files?.[0])
  }

  return (
    <div className="w-full">
      <div className="mb-1.5 flex items-center gap-1.5">
        <Icon className="h-3.5 w-3.5 text-teal-500" strokeWidth={2} />
        <label className="text-[13px] font-medium text-slate-300">{field.label}</label>
        <span className="ml-auto rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-teal-400">
          {field.short}
        </span>
      </div>

      {!file ? (
        <div
          onDragOver={(event) => {
            event.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`relative flex items-center gap-3 rounded-lg border border-dashed px-3 py-3 transition-colors ${
            isDragging
              ? 'border-teal-400 bg-teal-400/5'
              : 'border-slate-700/80 bg-slate-900/40 hover:border-slate-600 hover:bg-slate-900/70'
          }`}
        >
          <UploadCloud className="h-4 w-4 shrink-0 text-slate-500" />
          <p className="text-xs text-slate-500">
            <span className="font-medium text-teal-400">Browse</span> or drop .nc file
          </p>
          <input
            type="file"
            accept=".nc"
            onChange={(event) => validateAndSet(event.target.files?.[0])}
            className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
            aria-label={`Upload ${field.label}`}
          />
        </div>
      ) : (
        <div className="flex items-center justify-between gap-2 rounded-lg border border-teal-800/40 bg-teal-950/20 px-3 py-2.5">
          <div className="flex min-w-0 items-center gap-2">
            <FileCheck2 className="h-4 w-4 shrink-0 text-teal-400" />
            <div className="min-w-0">
              <p className="truncate text-xs text-slate-200">{file.name}</p>
              <p className="text-[10px] text-slate-500">{formatSize(file.size)}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => onRemove(field.key)}
            className="shrink-0 rounded-md p-1 text-slate-500 transition-colors hover:bg-slate-800 hover:text-red-400"
            aria-label={`Remove ${field.label}`}
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {error && (
        <p className="mt-1.5 flex items-center gap-1 text-[11px] text-red-400">
          <AlertTriangle className="h-3 w-3" />
          {error}
        </p>
      )}
    </div>
  )
}
