export type LatLng = { lat: number; lng: number };
export type MissionPointDraft = { lat: string; lng: string };

export const isValidLat = (v: number) => Number.isFinite(v) && v >= -90 && v <= 90;
export const isValidLng = (v: number) => Number.isFinite(v) && v >= -180 && v <= 180;

export function parsePoint(draft: MissionPointDraft): LatLng | null {
  if (draft.lat.trim() === "" || draft.lng.trim() === "") return null;
  const lat = Number(draft.lat);
  const lng = Number(draft.lng);
  if (!isValidLat(lat) || !isValidLng(lng)) return null;
  return { lat, lng };
}

export const formatPoint = (p: LatLng | null) =>
  p ? `${p.lat.toFixed(4)}, ${p.lng.toFixed(4)}` : "Not selected";

export const toDraft = (p: LatLng): MissionPointDraft => ({
  lat: p.lat.toFixed(6),
  lng: p.lng.toFixed(6),
});

export const emptyDraft: MissionPointDraft = { lat: "", lng: "" };
export const MAX_RANGE_KM = 50;

export function haversineKm(a: LatLng, b: LatLng) {
  const R = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLng = ((b.lng - a.lng) * Math.PI) / 180;
  const la1 = (a.lat * Math.PI) / 180;
  const la2 = (b.lat * Math.PI) / 180;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(la1) * Math.cos(la2) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
