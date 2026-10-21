import { FilePlus2, FileText, MessageSquare, History } from "lucide-react";
import type { AnalysisSummary, View } from "../types";

interface Props {
  view: View;
  onView: (v: View) => void;
  history: AnalysisSummary[];
  currentId: string | null;
  onSelect: (id: string) => void;
}

const STATUS_DOT: Record<string, string> = { completed: "bg-emerald-500", error: "bg-red-500", processing: "bg-amber-500", pending: "bg-zinc-400" };

export default function Sidebar({ view, onView, history, currentId, onSelect }: Props) {
  const item = (v: View, label: string, Icon: typeof FileText) => (
    <button
      onClick={() => onView(v)}
      aria-current={view === v ? "page" : undefined}
      className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${view === v ? "bg-zinc-100 text-zinc-900" : "text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900"}`}
    >
      <Icon className="size-4 shrink-0" aria-hidden />
      {label}
    </button>
  );

  return (
    <aside className="shrink-0 border-b border-zinc-200 bg-white md:h-screen md:w-64 md:overflow-y-auto md:border-b-0 md:border-r">
      <div className="px-5 py-4 text-base font-semibold tracking-tight">Video Assistant</div>
      <nav className="flex gap-1 px-3 pb-3 md:flex-col md:pb-0" aria-label="Main">
        {item("new", "New Analysis", FilePlus2)}
        {item("results", "Results & Insights", FileText)}
        {item("chat", "RAG Chat", MessageSquare)}
      </nav>

      <details className="px-3 pb-3 md:mt-5 md:block" open>
        <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
          <History className="size-3.5" aria-hidden /> History
        </summary>
        {history.length === 0 ? (
          <p className="px-3 py-1 text-sm text-zinc-500">No analyses yet.</p>
        ) : (
          <ul className="max-h-60 space-y-0.5 overflow-y-auto md:max-h-none">
            {history.map((a) => (
              <li key={a.id}>
                <button
                  onClick={() => onSelect(a.id)}
                  className={`flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-left text-sm ${a.id === currentId ? "bg-zinc-100 font-medium" : "text-zinc-600 hover:bg-zinc-50"}`}
                  title={a.title ?? a.filename}
                >
                  <span className={`size-2 shrink-0 rounded-full ${STATUS_DOT[a.status]}`} aria-label={a.status} />
                  <span className="truncate">{a.title ?? a.filename}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
        <p className="px-3 pt-2 text-xs text-zinc-400">Kept in server memory; cleared when the backend restarts.</p>
      </details>
    </aside>
  );
}
