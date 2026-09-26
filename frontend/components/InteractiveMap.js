'use client'

import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useOceanStore, isLatValid, isLonValid } from '@/store/useOceanStore'
import { NIO_BOUNDS, NIO_CENTER } from '@/lib/constants'
import FloatingPanel from './FloatingPanel'

const pinIcon = L.divIcon({
  className: 'pin-marker',
  html: `<svg width="32" height="40" viewBox="0 0 32 40" xmlns="http://www.w3.org/2000/svg">
    <path d="M16 0C7.163 0 0 7.163 0 16c0 11 16 24 16 24s16-13 16-24c0-8.837-7.163-16-16-16z" fill="#fe4747ff"/>
    <circle cx="16" cy="16" r="6" fill="#022c22"/>
  </svg>`,
  iconSize: [32, 40],
  iconAnchor: [16, 40]
})

const southWest = [NIO_BOUNDS.latMin - 3, NIO_BOUNDS.lonMin - 3]
const northEast = [NIO_BOUNDS.latMax + 3, NIO_BOUNDS.lonMax + 3]
const nioBounds = [southWest, northEast]

function ClickCapture() {
  const setCoordinatesFromMap = useOceanStore((state) => state.setCoordinatesFromMap)

  useMapEvents({
    click(event) {
      const { lat, lng } = event.latlng
      const clampedLat = Math.min(Math.max(lat, NIO_BOUNDS.latMin), NIO_BOUNDS.latMax)
      const clampedLon = Math.min(Math.max(lng, NIO_BOUNDS.lonMin), NIO_BOUNDS.lonMax)
      setCoordinatesFromMap(clampedLat, clampedLon)
    }
  })

  return null
}

export default function InteractiveMap() {
  const latitude = useOceanStore((state) => state.latitude)
  const longitude = useOceanStore((state) => state.longitude)
  const hasPin = isLatValid(latitude) && isLonValid(longitude)

  return (
    <div className="w-full">
      <div className="relative h-[calc(100vh-13rem)] min-h-[520px] w-full overflow-hidden rounded-2xl border border-slate-800">
        <MapContainer
          center={NIO_CENTER}
          zoom={5}
          minZoom={4}
          maxZoom={10}
          maxBounds={nioBounds}
          maxBoundsViscosity={1.0}
          className="h-full w-full"
        >
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attribution='&copy; <a href="https://www.esri.com/">Esri</a>, Maxar, Earthstar Geographics, and the GIS User Community'
          />
          <ClickCapture />
          {hasPin && <Marker position={[Number(latitude), Number(longitude)]} icon={pinIcon} />}
        </MapContainer>

        <div className="pointer-events-none absolute right-4 top-4 z-[1000] hidden w-full max-w-sm sm:block">
          <div className="pointer-events-auto">
            <FloatingPanel />
          </div>
        </div>

        <div className="pointer-events-none absolute left-4 top-4 z-[1000] rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2 font-mono text-[11px] text-slate-400 backdrop-blur">
          North Indian Ocean &middot; {NIO_BOUNDS.latMin}&deg;N&ndash;{NIO_BOUNDS.latMax}&deg;N, {NIO_BOUNDS.lonMin}&deg;E&ndash;{NIO_BOUNDS.lonMax}&deg;E
        </div>
      </div>

      <div className="mt-4 sm:hidden">
        <FloatingPanel />
      </div>
    </div>
  )
}
