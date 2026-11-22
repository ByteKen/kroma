import { AnimatePresence, motion } from 'framer-motion';
import type { SkeletonWorkspaceView } from './workspaceTransition';

interface WorkspaceTransitionSkeletonProps {
  readonly workspace: SkeletonWorkspaceView | null;
}

function Skel({ className = '' }: { readonly className?: string }) {
  return <div className={`kroma-skeleton rounded-md ${className}`} />;
}

function HomeSkeleton() {
  return (
    <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden bg-canvas">
      <aside className="w-[248px] shrink-0 bg-surface-low border-r border-hairline p-4 flex flex-col gap-5">
        <div className="flex flex-col gap-2">
          <Skel className="h-3 w-20" />
          <Skel className="h-9 w-full rounded-lg" />
          <Skel className="h-9 w-full rounded-lg" />
          <Skel className="h-9 w-full rounded-lg" />
        </div>
        <div className="h-px bg-hairline" />
        <div className="flex flex-col gap-2">
          <Skel className="h-3 w-24" />
          <Skel className="h-16 w-full rounded-lg" />
        </div>
      </aside>

      <div className="flex-1 min-w-0 p-7 flex flex-col gap-6 overflow-hidden">
        <div className="flex items-center justify-between gap-4">
          <div className="flex flex-col gap-2">
            <Skel className="h-6 w-48" />
            <Skel className="h-3.5 w-72" />
          </div>
          <div className="flex items-center gap-2">
            <Skel className="h-8 w-28 rounded-lg" />
            <Skel className="h-8 w-32 rounded-lg" />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }, (_, i) => (
            <div
              key={i}
              className="rounded-xl bg-surface-low border border-hairline p-3 flex flex-col gap-3"
            >
              <div className="grid grid-cols-3 gap-1.5 h-[132px]">
                <Skel className="h-full rounded-lg" />
                <Skel className="h-full rounded-lg" />
                <Skel className="h-full rounded-lg" />
              </div>
              <div className="flex items-center justify-between gap-2">
                <Skel className="h-4 w-36" />
                <Skel className="h-4 w-14 rounded-full" />
              </div>
              <Skel className="h-3 w-48" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StudioDevelopSkeleton() {
  return (
    <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden bg-canvas">
      {/* Left Folders / Subjects Sidebar */}
      <aside className="w-[264px] shrink-0 bg-surface-low border-r border-hairline p-3 flex flex-col gap-2.5">
        <Skel className="h-8 w-full rounded-lg" />
        <Skel className="h-7 w-full rounded-md" />
        <div className="flex-1 flex flex-col gap-1.5 overflow-hidden pt-1">
          {Array.from({ length: 9 }, (_, i) => (
            <div
              key={i}
              className="h-[50px] shrink-0 rounded-lg bg-surface/55 border border-hairline/60 px-2 flex items-center gap-2.5"
            >
              <Skel className="w-8 h-9 rounded shrink-0" />
              <div className="flex-1 flex flex-col gap-1.5 min-w-0">
                <Skel className="h-3 w-24" />
                <Skel className="h-2.5 w-16" />
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* Center Stage: Toolbar + Photo Canvas + Filmstrip */}
      <div className="flex-1 min-w-0 flex flex-col overflow-hidden">
        <div className="h-11 px-4 bg-surface border-b border-hairline flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2">
            <Skel className="h-6 w-20" />
            <Skel className="h-4 w-36" />
          </div>
          <Skel className="h-7 w-52 rounded-lg" />
          <div className="flex items-center gap-2">
            <Skel className="h-6 w-16" />
            <Skel className="h-6 w-28" />
          </div>
        </div>

        <div className="flex-1 min-h-0 p-6 flex items-center justify-center">
          <Skel className="w-full max-w-[460px] h-full max-h-[560px] rounded-xl border border-hairline/50" />
        </div>

        <div className="h-[86px] shrink-0 bg-surface-low border-t border-hairline px-3 flex items-center gap-2 overflow-hidden">
          {Array.from({ length: 12 }, (_, i) => (
            <Skel key={i} className="w-[62px] h-[66px] rounded-md shrink-0" />
          ))}
        </div>
      </div>

      {/* Right Develop Inspector */}
      <aside className="w-[336px] shrink-0 bg-surface-low border-l border-hairline p-4 flex flex-col gap-4">
        <Skel className="h-[92px] w-full rounded-lg" />
        <Skel className="h-8 w-full rounded-lg" />
        <div className="flex flex-col gap-3.5 pt-1">
          {Array.from({ length: 8 }, (_, i) => (
            <div key={i} className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between">
                <Skel className="h-2.5 w-20" />
                <Skel className="h-2.5 w-10" />
              </div>
              <Skel className="h-1.5 w-full rounded-full" />
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}

function FaceOrganizerSkeleton() {
  return (
    <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden bg-canvas">
      {/* Left Subjects Column */}
      <aside className="w-[278px] shrink-0 bg-surface-low border-r border-hairline flex flex-col overflow-hidden">
        <div className="h-11 px-3 bg-surface border-b border-hairline flex items-center justify-between">
          <Skel className="h-4 w-24" />
          <Skel className="h-6 w-16" />
        </div>
        <div className="p-2.5 border-b border-hairline flex flex-col gap-2">
          <Skel className="h-7 w-full" />
          <div className="flex items-center gap-1.5">
            <Skel className="h-5 w-10" />
            <Skel className="h-5 w-14" />
            <Skel className="h-5 w-12" />
          </div>
        </div>
        <div className="flex-1 p-2.5 flex flex-col gap-2 overflow-hidden">
          {Array.from({ length: 8 }, (_, i) => (
            <div
              key={i}
              className="p-2 rounded-lg bg-surface/55 border border-hairline/60 flex items-center gap-2.5 shrink-0"
            >
              <Skel className="w-12 h-14 rounded shrink-0" />
              <div className="flex-1 flex flex-col gap-1.5">
                <Skel className="h-3.5 w-28" />
                <Skel className="h-2.5 w-20" />
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* Right Photo Grid */}
      <div className="flex-1 min-w-0 flex flex-col overflow-hidden relative">
        <div className="h-11 px-4 bg-surface border-b border-hairline flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2.5">
            <Skel className="h-4 w-32" />
            <Skel className="h-5 w-16 rounded-full" />
          </div>
          <div className="flex items-center gap-2">
            <Skel className="h-7 w-28 rounded-lg" />
            <Skel className="h-7 w-24 rounded-lg" />
          </div>
        </div>

        <div className="flex-1 p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3 content-start overflow-hidden">
          {Array.from({ length: 15 }, (_, i) => (
            <div
              key={i}
              className="rounded-lg bg-surface-low border border-hairline/60 p-2 flex flex-col gap-2"
            >
              <Skel className="w-full aspect-[4/5] rounded-md" />
              <div className="flex items-center justify-between">
                <Skel className="h-3 w-20" />
                <Skel className="h-3 w-10" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ExportSkeleton() {
  return (
    <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden bg-canvas">
      {/* Left Export Settings */}
      <aside className="w-[400px] shrink-0 bg-surface-low border-r border-hairline flex flex-col overflow-hidden">
        <div className="px-5 py-3.5 bg-surface border-b border-hairline flex items-center justify-between">
          <div className="flex flex-col gap-1.5">
            <Skel className="h-4 w-28" />
            <Skel className="h-3 w-48" />
          </div>
          <Skel className="h-5 w-20 rounded-full" />
        </div>
        <div className="flex-1 p-4 flex flex-col gap-4 overflow-hidden">
          <Skel className="h-24 w-full rounded-lg" />
          <Skel className="h-24 w-full rounded-lg" />
          <Skel className="h-28 w-full rounded-lg" />
          <Skel className="h-36 w-full rounded-lg" />
        </div>
        <div className="p-4 bg-surface border-t border-hairline flex flex-col gap-2.5">
          <Skel className="h-12 w-full rounded-lg" />
          <Skel className="h-10 w-full rounded-lg" />
        </div>
      </aside>

      {/* Right Output Preview Master-Detail */}
      <div className="flex-1 min-w-0 flex flex-col overflow-hidden">
        <div className="h-12 px-5 bg-surface border-b border-hairline flex items-center justify-between">
          <Skel className="h-4 w-56" />
          <div className="flex items-center gap-2">
            <Skel className="h-5 w-16 rounded-full" />
            <Skel className="h-5 w-24 rounded-full" />
          </div>
        </div>
        <div className="flex-1 flex min-h-0 overflow-hidden">
          <div className="flex-1 p-4 flex flex-col gap-2 overflow-hidden">
            {Array.from({ length: 9 }, (_, i) => (
              <div
                key={i}
                className="h-[60px] shrink-0 rounded-lg bg-surface/55 border border-hairline/60 px-3 flex items-center justify-between gap-3"
              >
                <div className="flex items-center gap-3">
                  <Skel className="w-9 h-10 rounded" />
                  <div className="flex flex-col gap-1.5">
                    <Skel className="h-3.5 w-32" />
                    <Skel className="h-2.5 w-24" />
                  </div>
                </div>
                <Skel className="h-5 w-16 rounded-full" />
              </div>
            ))}
          </div>
          <aside className="w-[330px] xl:w-[370px] shrink-0 bg-surface-low border-l border-hairline p-3 flex flex-col gap-2 overflow-hidden">
            <Skel className="h-12 w-full rounded-lg mb-1" />
            {Array.from({ length: 7 }, (_, i) => (
              <div
                key={i}
                className="p-2 rounded-lg bg-surface/55 border border-hairline/60 flex items-center gap-2.5 shrink-0"
              >
                <Skel className="w-11 h-13 rounded shrink-0" />
                <div className="flex-1 flex flex-col gap-1.5">
                  <Skel className="h-3 w-36" />
                  <Skel className="h-2.5 w-24" />
                </div>
              </div>
            ))}
          </aside>
        </div>
      </div>
    </div>
  );
}

function ReviewQueueSkeleton() {
  return (
    <div className="flex-1 flex min-w-0 min-h-0 overflow-hidden bg-canvas">
      {/* Left Verification Queue Sidebar */}
      <aside className="w-[300px] shrink-0 bg-surface-low border-r border-hairline flex flex-col overflow-hidden">
        <div className="h-11 px-3.5 bg-surface border-b border-hairline flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Skel className="h-4 w-28" />
            <Skel className="h-5 w-8 rounded-full" />
          </div>
          <Skel className="h-3 w-12" />
        </div>
        <div className="h-9 px-3 bg-surface-low border-b border-hairline flex items-center justify-between shrink-0">
          <Skel className="h-4 w-28" />
          <Skel className="h-4 w-12" />
        </div>
        <div className="flex-1 p-2.5 flex flex-col gap-2 overflow-hidden">
          {Array.from({ length: 7 }, (_, i) => (
            <div
              key={i}
              className="p-2 rounded-lg bg-surface/60 border border-hairline/60 flex items-center gap-2.5 shrink-0"
            >
              <Skel className="w-4 h-4 rounded shrink-0" />
              <Skel className="w-11 h-13 rounded shrink-0" />
              <div className="flex-1 flex flex-col gap-1.5 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <Skel className="h-3 w-20" />
                  <Skel className="h-3 w-8" />
                </div>
                <Skel className="h-2.5 w-24" />
                <Skel className="h-2.5 w-16" />
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* Right Side-by-Side Comparison Stage */}
      <div className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">
        <div className="h-13 px-5 bg-surface border-b border-hairline flex items-center justify-between shrink-0">
          <div className="flex flex-col gap-1.5">
            <Skel className="h-4 w-52" />
            <Skel className="h-2.5 w-64" />
          </div>
          <Skel className="h-5 w-24 rounded-full" />
        </div>

        <div className="flex-1 min-h-0 p-4 flex items-center justify-center overflow-hidden">
          <div className="grid grid-cols-2 gap-4 w-full max-w-[920px] h-full min-h-0">
            <Skel className="w-full h-full rounded-xl border border-hairline/50" />
            <Skel className="w-full h-full rounded-xl border border-hairline/50" />
          </div>
        </div>

        <div className="px-5 py-3 bg-surface border-t border-hairline flex flex-col gap-2.5 shrink-0">
          <Skel className="h-2 w-full max-w-[920px] mx-auto rounded-full" />
          <div className="flex items-center justify-center gap-2.5">
            <Skel className="h-9 w-52 rounded-lg" />
            <Skel className="h-9 w-52 rounded-lg" />
            <Skel className="h-9 w-36 rounded-lg" />
          </div>
        </div>
      </div>
    </div>
  );
}

function SlateOcrSkeleton() {
  return (
    <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden bg-canvas">
      <div className="h-12 px-5 bg-surface border-b border-hairline flex items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-2.5">
          <Skel className="h-4 w-24" />
          <Skel className="h-5 w-32 rounded-full" />
        </div>
        <div className="flex items-center gap-2">
          <Skel className="h-8 w-44 rounded-lg" />
          <Skel className="h-8 w-32 rounded-lg" />
        </div>
      </div>
      <div className="flex-1 flex min-h-0 overflow-hidden">
        <aside className="w-[420px] shrink-0 bg-surface-low border-r border-hairline p-3 flex flex-col gap-2 overflow-hidden">
          <Skel className="h-8 w-full rounded-md" />
          <div className="flex items-center gap-1.5">
            <Skel className="h-5 w-10" />
            <Skel className="h-5 w-14" />
            <Skel className="h-5 w-20" />
          </div>
          <div className="flex-1 flex flex-col gap-2 pt-1 overflow-hidden">
            {Array.from({ length: 8 }, (_, i) => (
              <div
                key={i}
                className="h-[74px] shrink-0 rounded-lg bg-surface/55 border border-hairline/60 px-2.5 flex items-center gap-2.5"
              >
                <Skel className="w-10 h-12 rounded shrink-0" />
                <div className="flex-1 flex flex-col gap-2">
                  <Skel className="h-3 w-32" />
                  <Skel className="h-6 w-full rounded" />
                </div>
              </div>
            ))}
          </div>
        </aside>
        <div className="flex-1 min-w-0 flex flex-col overflow-hidden">
          <div className="h-11 px-4 bg-surface border-b border-hairline flex items-center justify-between">
            <Skel className="h-4 w-40" />
            <div className="flex items-center gap-2">
              <Skel className="h-7 w-20 rounded-md" />
              <Skel className="h-7 w-24 rounded-md" />
            </div>
          </div>
          <div className="flex-1 p-6 flex items-center justify-center">
            <Skel className="w-full max-w-[440px] h-full max-h-[480px] rounded-xl border border-hairline/50" />
          </div>
          <div className="h-28 p-4 bg-surface border-t border-hairline flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <Skel className="h-9 flex-1 rounded-lg" />
              <Skel className="h-9 w-32 rounded-lg" />
              <Skel className="h-9 w-28 rounded-lg" />
            </div>
            <Skel className="h-4 w-72" />
          </div>
        </div>
      </div>
    </div>
  );
}

export function WorkspaceTransitionSkeleton({ workspace }: WorkspaceTransitionSkeletonProps) {
  return (
    <AnimatePresence>
      {workspace && (
        <motion.div
          key={`skeleton-${workspace}`}
          aria-hidden="true"
          initial={{ opacity: 1 }}
          animate={{ opacity: 1 }}
          exit={{
            opacity: 0,
            transition: { duration: 0.18, ease: [0.22, 1, 0.36, 1] },
          }}
          className="pointer-events-none absolute inset-0 z-30 flex min-w-0 min-h-0 overflow-hidden bg-canvas"
        >
          {workspace === 'home' && <HomeSkeleton />}
          {workspace === 'develop' && <StudioDevelopSkeleton />}
          {workspace === 'culling' && <FaceOrganizerSkeleton />}
          {workspace === 'slate_ocr' && <SlateOcrSkeleton />}
          {workspace === 'export' && <ExportSkeleton />}
          {workspace === 'review_queue' && <ReviewQueueSkeleton />}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
