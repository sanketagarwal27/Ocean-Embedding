# STAG-INR — Input UI

Frontend input interface for the STAG-INR ocean temperature prediction project. This phase covers **data collection only** — Manual form mode and Interactive map mode. No backend calls are wired up yet; submitting logs the payload to the browser console via the Zustand store.

## Stack

- Next.js 14 (App Router) + React 18
- Tailwind CSS
- Zustand (global state)
- React-Leaflet (map)
- Lucide React (icons)

Note: no semicolons are used anywhere in the `.js` source, per project style.

## Setup

```bash
npm install
npm run dev
```

Then open `http://localhost:3000`.

> `npm install` will flag `next@14.2.5` as having a known advisory. It builds and runs fine for this UI phase; bump to a patched 14.2.x release before shipping to production with `npm install next@latest`.

## Project structure

```
app/
  layout.js          Root layout, font + metadata
  page.js             Header, mode toggle, mode switcher
  globals.css         Ocean theme tokens, Leaflet dark-mode overrides
store/
  useOceanStore.js    Zustand store: mode, date, lat/lon, 7 files, submit logic
lib/
  constants.js        File field definitions + North Indian Ocean bounding box
components/
  ModeToggle.js       Manual / Map tab switch
  ManualInputForm.js  Full form: date, coordinates, 7 uploads, submit
  InteractiveMap.js   Bounded Leaflet map, click-to-pin
  FloatingPanel.js    Overlay panel used inside map mode (reuses FileUploadField)
  CoordinateInput.js  Numeric lat/lon field with bounding-box validation
  FileUploadField.js  Shared drag-and-drop .nc uploader used by both modes
```

## Behavior notes

- **Bounding box**: Latitude is constrained to 5°N–30°N, longitude to 45°E–105°E (North Indian Ocean). The manual form flags out-of-range values inline; the map hard-clamps clicks to the box and is bounded so users can't pan/zoom far outside it.
- **File validation**: each of the 7 uploaders only accepts `.nc` files and shows an inline error otherwise.
- **Submit button**: disabled until date + valid coordinates + all 7 files are present (`isReadyToSubmit()` in the store). On click it calls `submitPayload()`, which builds the payload object and `console.log`s it — swap this for your API call when the backend is ready.
- **State**: everything (mode, date, lat/lon, files) lives in one Zustand store so both input modes share state — e.g. switching from Map mode after dropping a pin keeps the coordinates if you flip to Manual mode.

## Wiring up the backend later

Replace the body of `submitPayload` in `store/useOceanStore.js` with your API/upload call — the payload shape (`observationDate`, `coordinates`, `files`) is already assembled there.
