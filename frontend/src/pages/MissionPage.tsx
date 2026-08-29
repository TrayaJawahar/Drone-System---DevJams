import { Link, useSearchParams, Navigate } from "react-router-dom";
import { lazy, Suspense } from "react";
import { z } from "zod";
import { Header } from "../components/Header";
import { MissionSummary } from "../components/MissionSummary";

const MissionMap = lazy(() => import("../components/MissionMap"));

const coord = (min: number, max: number) =>
  z.preprocess(
    (v) => (v === "" || v === undefined || v === null ? undefined : Number(v)),
    z.number().min(min).max(max).optional(),
  );

const searchSchema = z.object({
  startLatitude: coord(-90, 90),
  startLongitude: coord(-180, 180),
  destinationLatitude: coord(-90, 90),
  destinationLongitude: coord(-180, 180),
});

const modules = [
  { key: "situation", title: "Situation Analysis", detail: "Terrain, obstacles and no-fly zones" },
  { key: "network", title: "Network Quality", detail: "Signal coverage along the corridor" },
  { key: "rl", title: "RL Route Planning", detail: "Reward-optimised waypoint policy" },
  { key: "sim", title: "Drone Simulation", detail: "Flight playback and telemetry" },
];

export function MissionPage() {
  const [searchParams] = useSearchParams();
  const searchObj = Object.fromEntries(searchParams.entries());
  
  const parsed = searchSchema.safeParse(searchObj);
  
  if (!parsed.success) {
    return <Navigate to="/" replace />;
  }

  const s = parsed.data;
  
  const values = [
    s.startLatitude,
    s.startLongitude,
    s.destinationLatitude,
    s.destinationLongitude,
  ];
  
  if (values.some((v) => typeof v !== "number" || Number.isNaN(v))) {
    return <Navigate to="/" replace />;
  }

  const start = { lat: s.startLatitude ?? 0, lng: s.startLongitude ?? 0 };
  const destination = { lat: s.destinationLatitude ?? 0, lng: s.destinationLongitude ?? 0 };
  const searchString = searchParams.toString();

  return (
    <div className="min-h-screen bg-background pt-14">
      <Header />
      <main className="mx-auto max-w-6xl px-5 py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold tracking-[0.14em] text-foreground" style={{ margin: 0 }}>MISSION ANALYSIS</h1>
            <p className="mt-1 text-sm text-muted-foreground" style={{ margin: 0, marginTop: '4px' }}>
              Corridor locked. Analysis modules will attach to this mission context.
            </p>
          </div>
          <Link
            to={`/?${searchString}`}
            className="rounded-md border border-border/70 px-3 py-2 text-xs font-medium tracking-[0.14em] text-muted-foreground transition-colors hover:border-ring/60 hover:text-foreground"
          >
            ← EDIT COORDINATES
          </Link>
        </div>

        <div className="mt-6 overflow-hidden rounded-lg border border-border/70">
          <div className="relative h-[380px]">
            <Suspense fallback={<div className="grid size-full place-items-center text-sm text-muted-foreground">Loading map…</div>}>
              <MissionMap start={start} destination={destination} onMapClick={() => {}} />
            </Suspense>
            <div className="pointer-events-none absolute left-4 top-4 z-[500] rounded-md border border-border/70 bg-card/85 px-3 py-2 text-[11px] uppercase tracking-[0.16em] text-muted-foreground backdrop-blur-sm shadow-sm">
              Planned corridor · <span className="text-accent">start</span> →{" "}
              <span className="text-destination">destination</span>
            </div>
          </div>
        </div>

        <div className="mt-4 grid gap-4 lg:grid-cols-[320px_1fr]">
          <MissionSummary start={start} destination={destination} />
          <div className="grid gap-4 sm:grid-cols-2">
            {modules.map((m) => (
              <div key={m.key} className="rounded-lg border border-border/70 bg-card/60 p-4 backdrop-blur-sm">
                <h2 className="text-sm font-semibold text-foreground" style={{ margin: 0 }}>{m.title}</h2>
                <p className="mt-1 text-xs text-muted-foreground" style={{ margin: 0, marginTop: '4px' }}>{m.detail}</p>
                <span className="mt-4 inline-block rounded border border-accent/30 bg-accent/10 px-2 py-1 text-[10px] uppercase tracking-[0.16em] text-accent">
                  pending
                </span>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
