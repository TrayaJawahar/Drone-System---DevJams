import { Link } from "react-router-dom";

export function Header() {
  return (
    <header className="fixed inset-x-0 top-0 z-50 h-14 border-b border-border/70 bg-card/80 backdrop-blur-md">
      <div className="flex h-full items-center gap-3 px-5">
        <Link to="/" className="flex items-center gap-3">
          <span className="grid size-8 place-items-center rounded-md border border-accent/40 bg-accent/10 text-accent">
            <svg viewBox="0 0 24 24" className="size-4 w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.6">
              <circle cx="12" cy="12" r="2.6" />
              <path d="M12 9.4V6m0 12v-3.4M9.4 12H6m12 0h-3.4" />
              <circle cx="5" cy="5" r="2" />
              <circle cx="19" cy="5" r="2" />
              <circle cx="5" cy="19" r="2" />
              <circle cx="19" cy="19" r="2" />
            </svg>
          </span>
          <span className="text-[15px] font-bold tracking-[0.14em] text-foreground sm:text-base">
            SITUATION AWARE DRONE SYSTEM
          </span>
        </Link>
        <span className="ml-auto hidden text-[11px] font-medium tracking-[0.18em] text-muted-foreground md:block">
          GIS · NETWORK INTELLIGENCE · RL ROUTING
        </span>
      </div>
    </header>
  );
}
