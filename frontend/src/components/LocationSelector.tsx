import { CoordinateInput } from "./CoordinateInput";
import type { MissionPointDraft } from "@/lib/mission";
import { isValidLat, isValidLng } from "@/lib/mission";

type Props = {
  title: string;
  tone: "start" | "destination";
  active: boolean;
  draft: MissionPointDraft;
  onChange: (draft: MissionPointDraft) => void;
  onActivate: () => void;
};

export function LocationSelector({ title, tone, active, draft, onChange, onActivate }: Props) {
  const dot = tone === "start" ? "bg-accent" : "bg-destination";
  const latInvalid = draft.lat.trim() !== "" && !isValidLat(Number(draft.lat));
  const lngInvalid = draft.lng.trim() !== "" && !isValidLng(Number(draft.lng));

  return (
    <section
      onFocus={onActivate}
      onClick={onActivate}
      className={`rounded-lg border p-4 transition-colors ${
        active ? "border-ring/60 bg-secondary/60" : "border-border/70 bg-secondary/25"
      }`}
    >
      <div className="mb-3 flex items-center gap-2">
        <span className={`size-2 rounded-full ${dot}`} />
        <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-foreground">{title}</h3>
        {active && (
          <span className="ml-auto text-[10px] uppercase tracking-[0.14em] text-accent">
            click map to set
          </span>
        )}
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <CoordinateInput
          label="Latitude"
          placeholder="Enter latitude"
          value={draft.lat}
          invalid={latInvalid}
          onChange={(lat) => onChange({ ...draft, lat })}
        />
        <CoordinateInput
          label="Longitude"
          placeholder="Enter longitude"
          value={draft.lng}
          invalid={lngInvalid}
          onChange={(lng) => onChange({ ...draft, lng })}
        />
      </div>
    </section>
  );
}
