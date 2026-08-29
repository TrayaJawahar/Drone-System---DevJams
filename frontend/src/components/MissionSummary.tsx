import { formatPoint, haversineKm, MAX_RANGE_KM, type LatLng } from "@/lib/mission";

type Props = { start: LatLng | null; destination: LatLng | null };

export function MissionSummary({ start, destination }: Props) {
  const distanceKm = start && destination ? haversineKm(start, destination) : null;
  const outOfRange = distanceKm !== null && distanceKm > MAX_RANGE_KM;
  return (
    <div className="rounded-lg border border-border/70 bg-card/60 p-4 backdrop-blur-sm">
      <h4 className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
        Selected Mission
      </h4>
      <dl className="mt-3 space-y-2 text-sm">
        <div className="flex items-baseline justify-between gap-3">
          <dt className="text-muted-foreground">Start</dt>
          <dd className="font-mono text-accent">{formatPoint(start)}</dd>
        </div>
        <div className="flex items-baseline justify-between gap-3">
          <dt className="text-muted-foreground">Destination</dt>
          <dd className="font-mono text-destination">{formatPoint(destination)}</dd>
        </div>
        <div className="flex items-baseline justify-between gap-3 border-t border-border/60 pt-2">
          <dt className="text-muted-foreground">Direct distance</dt>
          <dd className="font-mono text-foreground">
            {distanceKm !== null ? `${distanceKm.toFixed(2)} km` : "—"}
          </dd>
        </div>
      </dl>
      {outOfRange && (
        <p role="alert" className="mt-3 rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          Not possible to fly that much — direct distance exceeds the {MAX_RANGE_KM} km
          operational range. Pick a closer destination.
        </p>
      )}
    </div>
  );
}
