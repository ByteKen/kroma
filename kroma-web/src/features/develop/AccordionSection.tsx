import { AnimatePresence, motion } from 'framer-motion';
import type { ReactNode } from 'react';
import { FAST_FADE } from '../darkroom-kit';

interface AccordionSectionProps {
  readonly title: string;
  readonly open: boolean;
  readonly onToggle: () => void;
  readonly modified?: boolean;
  readonly onReset?: () => void;
  readonly rightSlot?: ReactNode;
  readonly children: ReactNode;
}

/**
 * Lightroom Classic-style collapsible Develop panel header + body.
 * Keeps advanced controls tucked cleanly out of sight until opened, with a subtle amber indicator
 * whenever parameters inside the section are modified.
 */
export function AccordionSection({
  title,
  open,
  onToggle,
  modified = false,
  onReset,
  rightSlot,
  children,
}: AccordionSectionProps) {
  return (
    <section className="border-b border-hairline/80 last:border-b-0">
      <div className="flex items-center justify-between gap-2 px-3 py-2.5 hover:bg-surface/60 transition-colors select-none">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          className="flex-1 flex items-center gap-2 text-left min-w-0 cursor-pointer"
        >
          <span
            className={`material-symbols-outlined text-[16px] text-ink-telemetry transition-transform duration-150 ${
              open ? 'rotate-90 text-ink-primary' : ''
            }`}
          >
            chevron_right
          </span>
          <span className="font-display text-[12px] font-semibold tracking-tight text-ink-primary truncate">
            {title}
          </span>
          {modified && (
            <span
              className="w-1.5 h-1.5 rounded-full bg-amber-core shrink-0"
              title={`${title} has active adjustments`}
            />
          )}
        </button>

        <div className="flex items-center gap-1.5 shrink-0">
          {rightSlot}
          {modified && onReset && (
            <button
              type="button"
              onClick={onReset}
              title={`Reset ${title}`}
              className="tabular-nums-mono text-[9.5px] text-ink-telemetry hover:text-amber-soft px-1 py-0.5 rounded transition-colors cursor-pointer"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={FAST_FADE}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3.5 pt-1 flex flex-col gap-2.5">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}
