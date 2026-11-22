import { motion } from 'framer-motion';
import { TACTILE_SPRING } from './motionPresets';

interface DarkroomToggleProps {
  readonly checked: boolean;
  readonly onChange: (checked: boolean) => void;
  readonly label?: string;
  readonly ariaLabel?: string;
  readonly className?: string;
}

export function DarkroomToggle({
  checked,
  onChange,
  label = '',
  ariaLabel,
  className = '',
}: DarkroomToggleProps) {
  return (
    <label className={`flex items-center justify-between gap-2 cursor-pointer ${className}`}>
      {label && (
        <span className="text-[11px] text-ink-secondary leading-tight select-none">{label}</span>
      )}
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={ariaLabel ?? (label || undefined)}
        onClick={() => onChange(!checked)}
        className={`relative w-8 h-[18px] rounded-full p-[2px] flex items-center transition-colors cursor-pointer shrink-0 ${
          checked ? 'bg-amber-core justify-end' : 'bg-surface-highest justify-start'
        }`}
      >
        <motion.span
          layout
          transition={TACTILE_SPRING}
          className="w-3.5 h-3.5 rounded-full bg-canvas shadow-sm block"
        />
      </button>
    </label>
  );
}
