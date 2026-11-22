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
      className={`inline-flex items-center bg-canvas rounded-lg p-0.5 border border-hairline ${className}`}
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
              className={`relative px-2 py-1 rounded-md text-[11px] flex items-center gap-1 transition-colors cursor-pointer ${
                isActive ? 'text-amber-soft font-medium' : 'text-ink-telemetry hover:text-ink-primary'
              }`}
            >
              {isActive && (
                <motion.span
                  layoutId={layoutId}
                  transition={TACTILE_SPRING}
                  className="absolute inset-0 rounded-md bg-surface-active border border-hairline-strong"
                />
              )}
              <span className="relative z-10 flex items-center gap-1">
                {option.icon && (
                  <span className="material-symbols-outlined text-[16px]">{option.icon}</span>
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
