import { useEffect, useRef } from "react";
import L from "leaflet";
import type { LatLng } from "../lib/mission";

type Props = {
  start: LatLng | null;
  destination: LatLng | null;
  onMapClick: (point: LatLng) => void;
};

const startIcon = L.divIcon({
  className: "",
  html: `<div class="marker-pin marker-pin--start"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="3"/><path d="M12 9V5M12 19v-4M9 12H5m14 0h-4"/></svg></div>`,
  iconSize: [30, 30],
  iconAnchor: [15, 15],
});

const destIcon = L.divIcon({
  className: "",
  html: `<div class="marker-pin marker-pin--dest"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2" fill="currentColor" stroke="none"/><path d="M12 1v3m0 16v3M1 12h3m16 0h3"/></svg></div>`,
  iconSize: [30, 30],
  iconAnchor: [15, 15],
});

export default function MissionMap({ start, destination, onMapClick }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const startMarker = useRef<L.Marker | null>(null);
  const destMarker = useRef<L.Marker | null>(null);
  const line = useRef<L.Polyline | null>(null);
  const clickRef = useRef(onMapClick);
  clickRef.current = onMapClick;

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, {
      center: [13.0827, 80.2707],
      zoom: 12,
      zoomControl: true,
      attributionControl: true,
    });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
    map.on("click", (e: L.LeafletMouseEvent) =>
      clickRef.current({ lat: e.latlng.lat, lng: e.latlng.lng }),
    );
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const sync = (
      ref: React.MutableRefObject<L.Marker | null>,
      point: LatLng | null,
      icon: L.DivIcon,
    ) => {
      if (!point) {
        ref.current?.remove();
        ref.current = null;
        return;
      }
      if (ref.current) ref.current.setLatLng(point);
      else ref.current = L.marker(point, { icon }).addTo(map);
    };

    sync(startMarker, start, startIcon);
    sync(destMarker, destination, destIcon);

    line.current?.remove();
    line.current = null;
    if (start && destination) {
      line.current = L.polyline([start, destination], {
        color: "#22d3ee",
        weight: 2.5,
        opacity: 0.85,
        dashArray: "8 8",
      }).addTo(map);
      map.fitBounds(L.latLngBounds([start, destination]).pad(0.35));
    } else if (start || destination) {
      map.panTo((start ?? destination) as LatLng);
    }
  }, [start, destination]);

  return <div ref={containerRef} className="size-full" />;
}
