import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, type ReactNode } from 'react';
import { FAST_FADE, SOFT_SPRING } from './motionPresets';

interface DarkroomModalProps {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly title: string;
  readonly subtitle?: string;
  readonly icon?: string;
  readonly maxWidthClass?: string;
  readonly footer?: ReactNode;
  readonly children: ReactNode;
}

export function DarkroomModal({
  open,
  onClose,
  title,
  subtitle,
  icon,
  maxWidthClass = 'max-w-lg',
  footer,
  children,
}: DarkroomModalProps) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={FAST_FADE}
            onClick={onClose}
            className="absolute inset-0 bg-canvas/80 backdrop-blur-sm"
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: 6 }}
            transition={SOFT_SPRING}
            className={`relative z-10 w-full ${maxWidthClass} rounded-xl bg-surface border border-hairline-strong shadow-[0_12px_40px_rgba(0,0,0,0.75)] overflow-hidden flex flex-col`}
          >
            <div className="px-4 py-3 bg-surface-low border-b border-hairline flex items-center justify-between gap-3">
              <div className="flex items-center gap-2.5 min-w-0">
                {icon && (
                  <span className="material-symbols-outlined text-[18px] text-amber-soft shrink-0">
                    {icon}
                  </span>
                )}
                <div className="min-w-0">
                  <h2 className="font-display text-[14px] font-bold text-ink-primary truncate">
                    {title}
                  </h2>
                  {subtitle && (
                    <p className="text-[11px] text-ink-secondary truncate">{subtitle}</p>
                  )}
                </div>
              </div>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close dialog"
                className="w-7 h-7 rounded-md text-ink-telemetry hover:bg-surface-active hover:text-ink-primary flex items-center justify-center transition-colors cursor-pointer"
              >
                <span className="material-symbols-outlined text-[16px]">close</span>
              </button>
            </div>

            <div className="p-4 overflow-y-auto max-h-[70vh]">{children}</div>

            {footer && (
              <div className="px-4 py-3 bg-surface-low border-t border-hairline flex items-center justify-end gap-2">
                {footer}
              </div>
            )}
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
