import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Loader2, MessageSquare, SendHorizonal } from "lucide-react";
import FormattedReport from "./FormattedReport";
import { askQuestion } from "../services/api";
import type { Analysis, ChatMessage } from "../types";

interface Props {
  analysis: Analysis | null;
  messages: ChatMessage[];
  setMessages: (updater: (prev: ChatMessage[]) => ChatMessage[]) => void;
}

let nextId = 1;

export default function RagChat({ analysis, messages, setMessages }: Props) {
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [messages, pending]);

  if (!analysis) {
    return (
      <div className="mx-auto mt-16 max-w-md text-center">
        <MessageSquare className="mx-auto size-8 text-zinc-300" aria-hidden />
        <h2 className="mt-3 text-lg font-semibold">No analysis selected</h2>
        <p className="mt-1 text-sm text-zinc-600">Upload a file or choose one from History to chat with it.</p>
      </div>
    );
  }

  const ready = analysis.status === "completed";

  async function send() {
    const q = draft.trim();
    if (!q || !ready || pending || !analysis) return;
    setDraft("");
    setPending(true);
    setMessages((m) => [...m, { id: nextId++, role: "user", text: q }]);
    try {
      const answer = await askQuestion(analysis.id, q);
      setMessages((m) => [...m, { id: nextId++, role: "assistant", text: answer }]);
    } catch (e) {
      setMessages((m) => [...m, { id: nextId++, role: "assistant", text: (e as Error).message, isError: true }]);
    } finally {
      setPending(false);
    }
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="mx-auto flex h-[calc(100vh-7rem)] max-w-3xl flex-col md:h-[calc(100vh-4rem)]">
      <div className="mb-3">
        <h1 className="text-xl font-semibold tracking-tight">RAG Chat</h1>
        <p className="truncate text-sm text-zinc-500">{analysis.title ?? analysis.filename}</p>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto rounded-xl border border-zinc-200 p-4" role="log" aria-live="polite">
        {!ready && <p className="text-sm text-zinc-600">Chat becomes available when the analysis is complete{analysis.status === "error" ? " (this one failed)" : ""}.</p>}
        {ready && messages.length === 0 && <p className="text-sm text-zinc-500">Ask anything about this recording. Answers come only from its transcript.</p>}
        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="flex justify-end"><div className="max-w-[85%] whitespace-pre-wrap rounded-2xl bg-zinc-900 px-4 py-2 text-sm text-white">{m.text}</div></div>
          ) : (
            <div key={m.id} className={`max-w-[90%] rounded-2xl border px-4 py-2 text-sm ${m.isError ? "border-red-200 bg-red-50 text-red-800" : "border-zinc-200 bg-white"}`}>
              {m.isError ? m.text : <FormattedReport markdown={m.text} />}
            </div>
          ),
        )}
        {pending && <div className="flex items-center gap-2 text-sm text-zinc-500"><Loader2 className="size-4 animate-spin" aria-hidden />Thinking…</div>}
        <div ref={end} />
      </div>

      <div className="mt-3 flex items-end gap-2">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKey}
          rows={2}
          maxLength={2000}
          disabled={!ready}
          aria-label="Your question"
          placeholder={ready ? "Ask a question… (Enter to send, Shift+Enter for a new line)" : "Waiting for the analysis to finish…"}
          className="flex-1 resize-none rounded-lg border border-zinc-300 px-3.5 py-2 text-sm disabled:bg-zinc-50"
        />
        <button onClick={send} disabled={!ready || pending || !draft.trim()} aria-label="Send" className="inline-flex h-10 items-center gap-2 rounded-lg bg-zinc-900 px-4 text-sm font-medium text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-300">
          <SendHorizonal className="size-4" aria-hidden />Send
        </button>
      </div>
    </div>
  );
}
