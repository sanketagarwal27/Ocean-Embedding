'use client'

import dynamic from 'next/dynamic'
import { Waves } from 'lucide-react'
import { useOceanStore } from '@/store/useOceanStore'
import ModeToggle from '@/components/ModeToggle'
import ManualInputForm from '@/components/ManualInputForm'

const InteractiveMap = dynamic(() => import('@/components/InteractiveMap'), {
  ssr: false,
  loading: () => (
    <div className="flex h-[calc(100vh-13rem)] min-h-[520px] items-center justify-center rounded-2xl border border-slate-800 bg-slate-900/40">
      <p className="text-sm text-slate-500">Loading map&hellip;</p>
    </div>
  )
})

export default function Home() {
  const mode = useOceanStore((state) => state.mode)

  return (
    <main className="min-h-screen px-4 py-10 sm:px-8">
      <div className="mx-auto mb-8 flex max-w-6xl flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-teal-600 to-cyan-800 shadow-glow">
            <Waves className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="font-display text-xl font-semibold text-slate-100">STAG&#8209;INR</h1>
            <p className="text-xs text-slate-500">Subsurface Temperature Grid &middot; North Indian Ocean</p>
          </div>
        </div>

        <ModeToggle />
      </div>

      <div className="mx-auto max-w-6xl">{mode === 'manual' ? <ManualInputForm /> : <InteractiveMap />}</div>
    </main>
  )
}
