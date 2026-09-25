import { create } from 'zustand'
import { EMPTY_FILES, NIO_BOUNDS } from '@/lib/constants'

export const isLatValid = (lat) =>
  lat !== '' && !Number.isNaN(Number(lat)) && Number(lat) >= NIO_BOUNDS.latMin && Number(lat) <= NIO_BOUNDS.latMax

export const isLonValid = (lon) =>
  lon !== '' && !Number.isNaN(Number(lon)) && Number(lon) >= NIO_BOUNDS.lonMin && Number(lon) <= NIO_BOUNDS.lonMax

export const useOceanStore = create((set, get) => ({
  mode: 'manual',
  date: '',
  latitude: '',
  longitude: '',
  files: { ...EMPTY_FILES },
  lastSubmission: null,
  
  // Backend Integration State
  predictionData: null,
  isPredicting: false,
  predictionError: null,

  setMode: (mode) => set({ mode }),

  setDate: (date) => set({ date }),

  setLatitude: (latitude) => set({ latitude }),

  setLongitude: (longitude) => set({ longitude }),

  setCoordinatesFromMap: (lat, lon) =>
    set({
      latitude: Number(lat.toFixed(4)),
      longitude: Number(lon.toFixed(4))
    }),

  setFile: (key, file) =>
    set((state) => ({
      files: { ...state.files, [key]: file }
    })),

  removeFile: (key) =>
    set((state) => ({
      files: { ...state.files, [key]: null }
    })),

  resetForm: () =>
    set({
      date: '',
      latitude: '',
      longitude: '',
      files: { ...EMPTY_FILES }
    }),

  isReadyToSubmit: () => {
    const { mode, date, latitude, longitude, files } = get()
    const hasValidRequestDetails = Boolean(date) && isLatValid(latitude) && isLonValid(longitude)

    if (mode === 'map') return hasValidRequestDetails

    const allFilesPresent = Object.values(files).every((f) => f !== null)
    return hasValidRequestDetails && allFilesPresent
  },

  submitPayload: async () => {
    const { mode, date, latitude, longitude, files } = get()

    const payload = {
      mode,
      observationDate: date,
      coordinates: { latitude, longitude },
      ...(mode === 'manual' && {
        files: Object.fromEntries(
          Object.entries(files).map(([key, file]) => [
            key,
            file ? { name: file.name, sizeBytes: file.size } : null
          ])
        )
      })
    }

    console.log('STAG-INR submission payload:', payload)
    set({ 
      lastSubmission: payload, 
      isPredicting: true, 
      predictionError: null, 
      predictionData: null 
    })
    
    // Call the actual FastAPI Backend
    try {
      const response = await fetch(`http://localhost:8000/api/predict_profile?lat=${latitude}&lon=${longitude}&date=${date}`)
      
      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || errData.error || 'Failed to fetch prediction from API')
      }
      
      const data = await response.json()
      console.log('Backend Prediction Received:', data)
      set({ predictionData: data, isPredicting: false })
      return data
    } catch (error) {
      console.error('FastAPI Integration Error:', error)
      set({ predictionError: error.message, isPredicting: false })
      return null
    }
  }
}))
