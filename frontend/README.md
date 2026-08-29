# Mission Pilot AI — Lovable project export

This ZIP contains the core source code exported from the Lovable project:

**SITUATION AWARE DRONE SYSTEM / Mission Pilot AI**

Project ID: `aca8ab7b-7a16-4e6e-9048-f3e89af5257f`

## Included

- Interactive Leaflet/OpenStreetMap mission map
- Start and destination coordinate selection
- Bidirectional map/input synchronization
- Coordinate validation
- 50 km operational-range validation
- Mission summary and direct-distance calculation
- `/mission` analysis page
- TanStack Router / TanStack Start configuration
- Tailwind CSS design system and technical dashboard styling

## Run

Requirements: Node.js 20+ (or a compatible Bun setup).

```bash
npm install
npm run dev
```

Then open the local URL printed by Vite.

The map uses OpenStreetMap tiles, so internet access is required for map tiles.

## Note

The ZIP intentionally focuses on the application-specific source and required configuration rather than the large set of unused generated shadcn/ui component files present in the Lovable repository.
