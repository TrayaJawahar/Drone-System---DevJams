type Props = {
  label: string;
  placeholder: string;
  value: string;
  onChange: (value: string) => void;
  invalid?: boolean;
};

export function CoordinateInput({ label, placeholder, value, onChange, invalid }: Props) {
  return (
    <label className="block">
      <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </span>
      <input
        inputMode="decimal"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        aria-invalid={invalid}
        className="mt-1.5 w-full rounded-md border border-input bg-background/60 px-3 py-2 font-mono text-sm text-foreground outline-none transition-colors placeholder:font-sans placeholder:text-muted-foreground/70 focus:border-ring focus:ring-2 focus:ring-ring/25 aria-[invalid=true]:border-destructive"
      />
    </label>
  );
}
