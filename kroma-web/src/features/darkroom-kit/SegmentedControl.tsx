import { motion } from 'framer-motion';
import { DarkroomTooltip, type TooltipSide } from './DarkroomTooltip';
import { TACTILE_SPRING } from './motionPresets';

export interface SegmentedOption<T extends string> {
  readonly id: T;
  readonly label?: string;
  readonly icon?: string;
  readonly title: string;
}

interface SegmentedControlProps<T extends string> {
  readonly layoutId: string;
  readonly options: readonly SegmentedOption<T>[];
  readonly value: T;
  readonly onChange: (next: T) => void;
  readonly tooltipSide?: TooltipSide;
  readonly className?: string;
}

export function SegmentedControl<T extends string>({
  layoutId,
  options,
  value,
  onChange,
  tooltipSide = 'auto',
  className = '',
}: SegmentedControlProps<T>) {
  return (
    <div
      role="radiogroup"
      className={`inline-flex items-center bg-surface-lowest/90 rounded-lg p-0.5 border border-hairline/70 ${className}`}
    >
      {options.map((option) => {
        const isActive = option.id === value;
        return (
          <DarkroomTooltip key={option.id} content={option.title} side={tooltipSide}>
            <button
              type="button"
              role="radio"
              aria-checked={isActive}
              onClick={() => onChange(option.id)}
              className={`relative px-2.5 py-1 rounded-md text-[11px] flex items-center justify-center gap-1 transition-colors duration-150 cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-amber-core/60 ${
                isActive ? 'text-ink-primary font-medium' : 'text-ink-telemetry hover:text-ink-secondary'
              }`}
            >
              {isActive && (
                <motion.span
                  layoutId={layoutId}
                  transition={TACTILE_SPRING}
                  className="absolute inset-0 rounded-md bg-surface-active border border-hairline-strong/80 shadow-[0_1px_2px_rgba(0,0,0,0.3)]"
                />
              )}
              <span className="relative z-10 flex items-center gap-1">
                {option.icon && (
                  <span className="material-symbols-outlined text-[15px]">{option.icon}</span>
                )}
                {option.label && <span>{option.label}</span>}
              </span>
            </button>
          </DarkroomTooltip>
        );
      })}
    </div>
  );
}
