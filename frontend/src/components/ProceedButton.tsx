type Props = {
  disabled?: boolean;
  onClick: () => void;
  label?: string;
};

export function ProceedButton({ disabled, onClick, label = "PROCEED →" }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="w-full rounded-md bg-gradient-accent px-4 py-3 text-sm font-semibold tracking-[0.16em] text-accent-foreground shadow-accent transition-all hover:brightness-110 active:scale-[0.99] disabled:cursor-not-allowed disabled:bg-none disabled:bg-secondary disabled:text-muted-foreground disabled:shadow-none"
    >
      {label}
    </button>
  );
}
