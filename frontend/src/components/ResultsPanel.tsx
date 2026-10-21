import { useState } from "react";
import { Check, Copy, Loader2, FileText } from "lucide-react";
import FormattedReport from "./FormattedReport";
import type { Analysis } from "../types";

const TABS = [
  { id: "summary", label: "Summary" },
  { id: "transcript", label: "Transcript" },
  { id: "action_items", label: "Action Items" },
  { id: "key_decisions", label: "Key Decisions" },
  { id: "open_questions", label: "Open Questions" },
] as const;
type TabId = (typeof TABS)[number]["id"];

export default function ResultsPanel({ analysis, loadError, onOpenChat, onNew }: { analysis: Analysis | null; loadError: string | null; onOpenChat: () => void; onNew: () => void }) {
  const [tab, setTab] = useState<TabId>("summary");
  const [copied, setCopied] = useState<"ok" | "fail" | null>(null);

  if (!analysis) {
    return (
      <div className="mx-auto mt-16 max-w-md text-center">
        <FileText className="mx-auto size-8 text-zinc-300" aria-hidden />
        <h2 className="mt-3 text-lg font-semibold">{loadError ? "Couldn't load the analysis" : "No analysis selected"}</h2>
        <p className="mt-1 text-sm text-zinc-600">{loadError ?? "Upload a file, or pick one from History."}</p>
        <button onClick={onNew} className="mt-4 rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800">New analysis</button>
      </div>
    );
  }

  if (analysis.status === "pending" || analysis.status === "processing") {
    return (
      <div className="mx-auto mt-16 max-w-md text-center" role="status">
        <Loader2 className="mx-auto size-7 animate-spin text-zinc-500" aria-hidden />
        <h2 className="mt-3 text-lg font-semibold">{analysis.status === "pending" ? "Waiting to start…" : "Analyzing…"}</h2>
        <p className="mt-1 text-sm text-zinc-600">{analysis.stage ?? "Queued"} · {analysis.filename}</p>
        <p className="mt-3 text-xs text-zinc-500">This can take several minutes for long recordings. Keep this page open.</p>
        {loadError && <p className="mt-3 text-sm text-red-700">{loadError}</p>}
      </div>
    );
  }

  if (analysis.status === "error") {
    return (
      <div className="mx-auto mt-16 max-w-md text-center">
        <h2 className="text-lg font-semibold">Analysis failed</h2>
        <p role="alert" className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-sm text-red-800">{analysis.error ?? "Something went wrong."}</p>
        <button onClick={onNew} className="mt-4 rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800">Try another file</button>
      </div>
    );
  }

  const body = analysis[tab] ?? "";
  async function copy() {
    try {
      await navigator.clipboard.writeText(body);
      setCopied("ok");
    } catch {
      setCopied("fail");
    }
    setTimeout(() => setCopied(null), 1800);
  }

  return (
    <div className="mx-auto max-w-3xl">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight">{analysis.title ?? analysis.filename}</h1>
          <p className="mt-1 truncate text-sm text-zinc-500">{analysis.filename} · {analysis.language}</p>
        </div>
        <button onClick={onOpenChat} className="rounded-lg border border-zinc-300 px-3.5 py-2 text-sm font-medium hover:bg-zinc-50">Chat with this recording</button>
      </div>

      <div role="tablist" aria-label="Results" className="mt-6 flex gap-1 overflow-x-auto border-b border-zinc-200">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`-mb-px whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium ${tab === t.id ? "border-zinc-900 text-zinc-900" : "border-transparent text-zinc-500 hover:text-zinc-800"}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div role="tabpanel" className="mt-5 rounded-xl border border-zinc-200 p-5">
        <div className="mb-3 flex justify-end">
          <button onClick={copy} className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium text-zinc-600 hover:bg-zinc-100">
            {copied === "ok" ? <Check className="size-3.5 text-emerald-600" /> : <Copy className="size-3.5" />}
            {copied === "ok" ? "Copied" : copied === "fail" ? "Copy failed" : "Copy"}
          </button>
        </div>
        {!body ? (
          <p className="text-sm text-zinc-500">Nothing to show.</p>
        ) : tab === "transcript" ? (
          <div className="max-h-[60vh] overflow-y-auto whitespace-pre-wrap break-words text-[15px] leading-7 text-zinc-800">{body}</div>
        ) : (
          <FormattedReport markdown={body} />
        )}
      </div>
    </div>
  );
}
