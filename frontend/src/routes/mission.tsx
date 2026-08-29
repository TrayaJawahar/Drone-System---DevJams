import { createFileRoute, Link, redirect, ClientOnly } from "@tanstack/react-router";
import { lazy, Suspense } from "react";
import { z } from "zod";
import { Header } from "@/components/Header";
import { MissionSummary } from "@/components/MissionSummary";

const MissionMap = lazy(() => import("@/components/MissionMap"));

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

export const Route = createFileRoute("/mission")({
  validateSearch: searchSchema,
  beforeLoad: ({ search }) => {
    const values = [
      search.startLatitude,
      search.startLongitude,
      search.destinationLatitude,
      search.destinationLongitude,
    ];
    if (values.some((v) => typeof v !== "number" || Number.isNaN(v))) {
      throw redirect({ to: "/" });
    }
  },
  head: () => ({
    meta: [
      { title: "Mission Analysis — Situation Aware Drone System" },
      {
        name: "description",
        content:
          "Situation analysis, network-quality assessment and RL route planning for the selected drone mission corridor.",
      },
    ],
  }),
  component: MissionPage,
});

const modules = [
  { key: "situation", title: "Situation Analysis", detail: "Terrain, obstacles and no-fly zones" },
  { key: "network", title: "Network Quality", detail: "Signal coverage along the corridor" },
  { key: "rl", title: "RL Route Planning", detail: "Reward-optimised waypoint policy" },
  { key: "sim", title: "Drone Simulation", detail: "Flight playback and telemetry" },
];

function MissionPage() {
  const s = Route.useSearch();
  const start = { lat: s.startLatitude ?? 0, lng: s.startLongitude ?? 0 };
  const destination = { lat: s.destinationLatitude ?? 0, lng: s.destinationLongitude ?? 0 };

  return (
    <div className="min-h-screen bg-background pt-14">
      <Header />
      <main className="mx-auto max-w-6xl px-5 py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold tracking-[0.14em] text-foreground">MISSION ANALYSIS</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Corridor locked. Analysis modules will attach to this mission context.
            </p>
          </div>
          <Link
            to="/"
            search={s}
            className="rounded-md border border-border/70 px-3 py-2 text-xs font-medium tracking-[0.14em] text-muted-foreground transition-colors hover:border-ring/60 hover:text-foreground"
          >
            ← EDIT COORDINATES
          </Link>
        </div>

        <div className="mt-6 overflow-hidden rounded-lg border border-border/70">
          <div className="relative h-[380px]">
            <ClientOnly fallback={<div className="grid size-full place-items-center text-sm text-muted-foreground">Loading map…</div>}>
              <Suspense fallback={<div className="grid size-full place-items-center text-sm text-muted-foreground">Loading map…</div>}>
                <MissionMap start={start} destination={destination} onMapClick={() => {}} />
              </Suspense>
            </ClientOnly>
            <div className="pointer-events-none absolute left-4 top-4 z-[500] rounded-md border border-border/70 bg-card/85 px-3 py-2 text-[11px] uppercase tracking-[0.16em] text-muted-foreground backdrop-blur-sm">
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
                <h2 className="text-sm font-semibold text-foreground">{m.title}</h2>
                <p className="mt-1 text-xs text-muted-foreground">{m.detail}</p>
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
