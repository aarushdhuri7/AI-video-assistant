import { useCallback, useEffect, useRef, useState } from "react";
import Sidebar from "./components/Sidebar";
import UploadPanel from "./components/UploadPanel";
import ResultsPanel from "./components/ResultsPanel";
import RagChat from "./components/RagChat";
import { ApiError, listAnalyses, pollAnalysis, uploadFile } from "./services/api";
import type { Analysis, AnalysisSummary, ChatMessage, Language, View } from "./types";

export default function App() {
  const [view, setView] = useState<View>("new");
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [current, setCurrent] = useState<Analysis | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [chats, setChats] = useState<Record<string, ChatMessage[]>>({});

  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const uploadCtrl = useRef<AbortController | null>(null);

  const refreshHistory = useCallback(() => {
    listAnalyses().then(setHistory).catch(() => { /* sidebar list is optional */ });
  }, []);
  useEffect(refreshHistory, [refreshHistory]);

  // Load/poll whichever analysis is selected; stop when the selection changes.
  useEffect(() => {
    if (!currentId) return;
    const ctrl = new AbortController();
    setCurrent(null);
    setLoadError(null);
    pollAnalysis(currentId, setCurrent, ctrl.signal)
      .then(refreshHistory)
      .catch((e) => {
        if (ctrl.signal.aborted) return;
        setLoadError(e instanceof ApiError && e.status === 404 ? "That analysis is no longer on the server (it is cleared when the backend restarts)." : (e as Error).message);
      });
    return () => ctrl.abort();
  }, [currentId, refreshHistory]);

  async function submit(file: File, language: Language) {
    setUploadError(null);
    setUploading(true);
    setProgress(0);
    uploadCtrl.current = new AbortController();
    try {
      const created = await uploadFile(file, language, setProgress, uploadCtrl.current.signal);
      setCurrentId(created.id);
      setView("results");
      refreshHistory();
    } catch (e) {
      if ((e as Error).name !== "AbortError") setUploadError((e as Error).message);
    } finally {
      setUploading(false);
      setProgress(null);
    }
  }

  const setMessages = (updater: (prev: ChatMessage[]) => ChatMessage[]) => {
    if (currentId) setChats((c) => ({ ...c, [currentId]: updater(c[currentId] ?? []) }));
  };

  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <Sidebar
        view={view}
        onView={setView}
        history={history}
        currentId={currentId}
        onSelect={(id) => { setCurrentId(id); setView("results"); }}
      />
      <main className="min-w-0 flex-1 p-5 md:h-screen md:overflow-y-auto md:p-8">
        {view === "new" && <UploadPanel busy={uploading} progress={progress} error={uploadError} onSubmit={submit} onCancel={() => uploadCtrl.current?.abort()} />}
        {view === "results" && <ResultsPanel key={currentId ?? "none"} analysis={current} loadError={loadError} onOpenChat={() => setView("chat")} onNew={() => setView("new")} />}
        {view === "chat" && <RagChat analysis={current} messages={currentId ? chats[currentId] ?? [] : []} setMessages={setMessages} />}
      </main>
    </div>
  );
}
