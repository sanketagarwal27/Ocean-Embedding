import { Thermometer, Droplets, Waves, ArrowRightLeft, ArrowUpDown, Wind, MoveVertical } from 'lucide-react'

export const NIO_BOUNDS = {
  latMin: 5,
  latMax: 30,
  lonMin: 45,
  lonMax: 105
}

export const NIO_CENTER = [17.5, 75]

export const FILE_FIELDS = [
  {
    key: 'sst',
    label: 'Sea Surface Temperature',
    short: 'SST',
    icon: Thermometer
  },
  {
    key: 'sss',
    label: 'Sea Surface Salinity',
    short: 'SSS',
    icon: Droplets
  },
  {
    key: 'ssh',
    label: 'Sea Surface Height',
    short: 'SSH',
    icon: Waves
  },
  {
    key: 'currentU',
    label: 'Surface Current U (Zonal)',
    short: 'U-Current',
    icon: ArrowRightLeft
  },
  {
    key: 'currentV',
    label: 'Surface Current V (Meridional)',
    short: 'V-Current',
    icon: ArrowUpDown
  },
  {
    key: 'windU',
    label: 'Surface Wind U (Zonal)',
    short: 'U-Wind',
    icon: Wind
  },
  {
    key: 'windV',
    label: 'Surface Wind V (Meridional)',
    short: 'V-Wind',
    icon: MoveVertical
  }
]

export const EMPTY_FILES = FILE_FIELDS.reduce((acc, field) => {
  acc[field.key] = null
  return acc
}, {})
