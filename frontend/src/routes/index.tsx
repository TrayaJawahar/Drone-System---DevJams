import { createFileRoute, useNavigate, ClientOnly } from "@tanstack/react-router";
import { lazy, Suspense, useMemo, useState } from "react";
import { Header } from "@/components/Header";
import { LocationSelector } from "@/components/LocationSelector";
import { MissionSummary } from "@/components/MissionSummary";
import { ProceedButton } from "@/components/ProceedButton";
import {
  emptyDraft,
  haversineKm,
  MAX_RANGE_KM,
  parsePoint,
  toDraft,
  type LatLng,
  type MissionPointDraft,
} from "@/lib/mission";

const MissionMap = lazy(() => import("@/components/MissionMap"));

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Mission Planner — Situation Aware Drone System" },
      {
        name: "description",
        content:
          "Plan an autonomous drone mission: pick start and destination coordinates on an interactive map for situation-aware, network-aware route planning.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  const navigate = useNavigate();
  const [startDraft, setStartDraft] = useState<MissionPointDraft>(emptyDraft);
  const [destDraft, setDestDraft] = useState<MissionPointDraft>(emptyDraft);
  const [active, setActive] = useState<"start" | "destination">("start");
  const [error, setError] = useState<string | null>(null);

  const start = useMemo(() => parsePoint(startDraft), [startDraft]);
  const destination = useMemo(() => parsePoint(destDraft), [destDraft]);
  const distanceKm = start && destination ? haversineKm(start, destination) : null;
  const outOfRange = distanceKm !== null && distanceKm > MAX_RANGE_KM;
  const ready = Boolean(start && destination) && !outOfRange;

  const handleMapClick = (point: LatLng) => {
    setError(null);
    if (active === "start") {
      setStartDraft(toDraft(point));
      setActive("destination");
    } else {
      setDestDraft(toDraft(point));
      setActive("start");
    }
  };

  const handleProceed = () => {
    if (!start || !destination) {
      setError("Please select both a start point and destination.");
      return;
    }
    if (haversineKm(start, destination) > MAX_RANGE_KM) {
      setError(
        `Not possible to fly that much — direct distance exceeds the ${MAX_RANGE_KM} km operational range.`,
      );
      return;
    }
    setError(null);
    navigate({
      to: "/mission",
      search: {
        startLatitude: start.lat,
        startLongitude: start.lng,
        destinationLatitude: destination.lat,
        destinationLongitude: destination.lng,
      },
    });
  };

  return (
    <div className="min-h-screen bg-background pt-14">
      <Header />
      <main className="grid h-[calc(100vh-3.5rem)] grid-cols-1 lg:grid-cols-[1fr_360px] xl:grid-cols-[1fr_400px]">
        <section className="relative min-h-[420px] border-b border-border/70 lg:border-b-0 lg:border-r">
          <ClientOnly fallback={<div className="grid size-full place-items-center text-sm text-muted-foreground">Loading map…</div>}>
            <Suspense fallback={<div className="grid size-full place-items-center text-sm text-muted-foreground">Loading map…</div>}>
              <MissionMap start={start} destination={destination} onMapClick={handleMapClick} />
            </Suspense>
          </ClientOnly>
          <div className="pointer-events-none absolute right-4 top-4 z-[500] rounded-md border border-border/70 bg-card/85 px-3 py-2 text-[11px] uppercase tracking-[0.16em] text-muted-foreground backdrop-blur-sm">
            Click map to set{" "}
            <span className={active === "start" ? "text-accent" : "text-destination"}>
              {active === "start" ? "start point" : "destination"}
            </span>
          </div>
        </section>

        <aside className="flex flex-col gap-4 overflow-y-auto bg-card/40 p-5 backdrop-blur-sm">
          <div>
            <h1 className="text-sm font-semibold uppercase tracking-[0.2em] text-foreground">
              Mission Coordinates
            </h1>
            <p className="mt-1 text-xs text-muted-foreground">
              WGS84 decimal degrees. Map and fields stay in sync.
            </p>
          </div>

          <LocationSelector
            title="Start Point"
            tone="start"
            active={active === "start"}
            draft={startDraft}
            onChange={(d) => {
              setStartDraft(d);
              setError(null);
            }}
            onActivate={() => setActive("start")}
          />
          <LocationSelector
            title="Destination"
            tone="destination"
            active={active === "destination"}
            draft={destDraft}
            onChange={(d) => {
              setDestDraft(d);
              setError(null);
            }}
            onActivate={() => setActive("destination")}
          />

          <MissionSummary start={start} destination={destination} />

          <div className="mt-auto space-y-2 pt-2">
            {error && <p role="alert" className="text-xs text-destructive">{error}</p>}
            <ProceedButton disabled={!ready} onClick={handleProceed} />
          </div>
        </aside>
      </main>
    </div>
  );
}
