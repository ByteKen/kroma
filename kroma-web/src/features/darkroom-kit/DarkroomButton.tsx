import { motion, type HTMLMotionProps } from 'framer-motion';
import type { ReactNode } from 'react';
import { DarkroomTooltip, useDarkroomTooltip, type TooltipSide } from './DarkroomTooltip';
import { TACTILE_SPRING } from './motionPresets';

export type DarkroomButtonVariant = 'primary' | 'secondary' | 'ghost' | 'emerald' | 'subtle';
export type DarkroomButtonSize = 'xs' | 'sm' | 'md';

interface DarkroomButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  readonly variant?: DarkroomButtonVariant;
  readonly size?: DarkroomButtonSize;
  readonly icon?: string;
  readonly iconClassName?: string;
  readonly active?: boolean;
  readonly tooltipSide?: TooltipSide;
  readonly children?: ReactNode;
}

const VARIANT_CLASSES: Record<DarkroomButtonVariant, string> = {
  primary:
    'bg-amber-core text-amber-ink font-display font-bold border border-amber-core hover:brightness-110',
  secondary:
    'bg-surface-elevated text-ink-primary border border-hairline hover:bg-surface-active hover:border-hairline-strong',
  subtle:
    'bg-canvas text-ink-secondary border border-hairline hover:bg-surface-active hover:text-ink-primary',
  ghost:
    'bg-transparent text-ink-secondary border border-transparent hover:bg-surface-active hover:text-ink-primary',
  emerald:
    'bg-emerald-focus/12 text-emerald-bright border border-emerald-focus/45 hover:bg-emerald-focus/20',
};

const ACTIVE_CLASSES =
  'bg-amber-core/15 text-amber-soft border border-amber-core/60 hover:bg-amber-core/20';

const SIZE_CLASSES: Record<DarkroomButtonSize, string> = {
  xs: 'h-7 px-2 text-[11px] rounded-md gap-1',
  sm: 'h-8 px-2.5 text-[11px] rounded-lg gap-1.5',
  md: 'h-8 px-3 text-[12px] rounded-lg gap-1.5',
};

export function DarkroomButton({
  variant = 'secondary',
  size = 'sm',
  icon,
  iconClassName = '',
  active = false,
  tooltipSide = 'auto',
  className = '',
  children,
  type = 'button',
  title,
  disabled,
  onMouseEnter,
  onMouseLeave,
  onFocus,
  onBlur,
  ...rest
}: DarkroomButtonProps) {
  const { onEnter, onLeave } = useDarkroomTooltip(title, tooltipSide);
  const visualClasses =
    active && variant !== 'primary' && variant !== 'emerald'
      ? ACTIVE_CLASSES
      : VARIANT_CLASSES[variant];

  const buttonNode = (
    <motion.button
      type={type}
      disabled={disabled}
      whileTap={{ scale: 0.97 }}
      transition={TACTILE_SPRING}
      onMouseEnter={(e) => {
        if (!disabled && title) onEnter(e.currentTarget);
        onMouseEnter?.(e);
      }}
      onMouseLeave={(e) => {
        if (!disabled && title) onLeave();
        onMouseLeave?.(e);
      }}
      onFocus={(e) => {
        if (!disabled && title) onEnter(e.currentTarget);
        onFocus?.(e);
      }}
      onBlur={(e) => {
        if (!disabled && title) onLeave();
        onBlur?.(e);
      }}
      className={`inline-flex items-center justify-center transition-colors cursor-pointer select-none disabled:opacity-45 disabled:pointer-events-none ${visualClasses} ${SIZE_CLASSES[size]} ${className}`}
      {...rest}
    >
      {icon && (
        <span className={`material-symbols-outlined text-[15px] shrink-0 ${iconClassName}`}>
          {icon}
        </span>
      )}
      {children}
    </motion.button>
  );

  // Disabled native buttons swallow mouse events; wrap cleanly so brief disabled guidance still displays
  if (disabled && title) {
    return (
      <DarkroomTooltip content={title} side={tooltipSide}>
        {buttonNode}
      </DarkroomTooltip>
    );
  }

  return buttonNode;
}
