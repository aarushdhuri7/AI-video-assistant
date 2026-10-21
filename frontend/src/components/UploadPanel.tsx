
import { useRef, useState, type DragEvent } from "react";
import { FileAudio, Loader2, Upload, X } from "lucide-react";
import type { Language } from "../types";

const EXTENSIONS = [
  "mp3",
  "wav",
  "m4a",
  "aac",
  "flac",
  "mp4",
  "mov",
  "mkv",
  "webm",
];

interface Props {
  busy: boolean;
  progress: number | null;
  error: string | null;
  onSubmit: (file: File, language: Language) => void;
  onCancel: () => void;
}

function size(n: number) {
  return n < 1024 ** 2
    ? `${(n / 1024).toFixed(0)} KB`
    : n < 1024 ** 3
      ? `${(n / 1024 ** 2).toFixed(1)} MB`
      : `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export default function UploadPanel({
  busy,
  progress,
  error,
  onSubmit,
  onCancel,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState<Language>("english");
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  function choose(f: File | undefined) {
    if (!f) return;

    const ext = f.name.split(".").pop()?.toLowerCase() ?? "";

    if (!EXTENSIONS.includes(ext)) {
      setFile(null);
      setLocalError(
        `Unsupported file type. Use one of: ${EXTENSIONS.join(", ")}.`
      );
      return;
    }

    setLocalError(null);
    setFile(f);
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);

    if (!busy) {
      choose(e.dataTransfer.files?.[0]);
    }
  }

  const shownError = localError ?? error;

  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="text-2xl font-semibold tracking-tight">
        New analysis
      </h1>

      <p className="mt-1 text-sm text-zinc-600">
        Upload a recording to get a transcript, summary, action items,
        decisions and open questions, then chat with it.
      </p>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`mt-6 rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
          dragging
            ? "border-zinc-900 bg-zinc-50"
            : "border-zinc-300"
        }`}
      >
        {file ? (
          <div className="flex items-center gap-3 text-left">
            <FileAudio
              className="size-6 shrink-0 text-zinc-500"
              aria-hidden
            />

            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">
                {file.name}
              </p>
              <p className="text-xs text-zinc-500">
                {size(file.size)}
              </p>
            </div>

            {!busy && (
              <button
                onClick={() => setFile(null)}
                className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-100"
                aria-label="Remove file"
              >
                <X className="size-4" />
              </button>
            )}
          </div>
        ) : (
          <>
            <Upload
              className="mx-auto size-7 text-zinc-400"
              aria-hidden
            />

            <p className="mt-2 text-sm text-zinc-700">
              Drag and drop a file here, or
            </p>

            <button
              onClick={() => input.current?.click()}
              className="mt-2 rounded-lg border border-zinc-300 px-3.5 py-1.5 text-sm font-medium hover:bg-zinc-50"
            >
              Browse files
            </button>
          </>
        )}

        <input
          ref={input}
          type="file"
          className="sr-only"
          accept={EXTENSIONS.map((e) => "." + e).join(",")}
          onChange={(e) => {
            choose(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>

      <p className="mt-2 text-xs text-zinc-500">
        Supported: {EXTENSIONS.join(", ")}. The maximum size is set
        by the server (500 MB by default).
      </p>

      {busy && progress !== null && (
        <div
          className="mt-4"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress * 100)}
          aria-label="Upload progress"
        >
          <div className="h-1.5 overflow-hidden rounded-full bg-zinc-200">
            <div
              className="h-full bg-zinc-900 transition-[width]"
              style={{ width: `${progress * 100}%` }}
            />
          </div>

          <p className="mt-1 text-xs text-zinc-500">
            Uploading… {Math.round(progress * 100)}%
          </p>
        </div>
      )}

      {shownError && (
        <p
          role="alert"
          className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-sm text-red-800"
        >
          {shownError}
        </p>
      )}

      <div className="mt-6 flex flex-wrap items-end justify-between gap-4">
        <label className="text-sm">
          <span className="mb-1 block font-medium text-zinc-700">
            Spoken language
          </span>

          <select
            value={language}
            onChange={(e) =>
              setLanguage(e.target.value as Language)
            }
            disabled={busy}
            className="h-10 rounded-lg border border-zinc-300 bg-white px-3 text-sm"
          >
            <option value="english">English</option>
            <option value="hindi">Hindi</option>
            <option value="auto">Auto-detect</option>
            <option value="hinglish">Hinglish (Sarvam AI)</option>
          </select>
        </label>

        <div className="flex gap-2">
          {busy && (
            <button
              onClick={onCancel}
              className="h-10 rounded-lg border border-zinc-300 px-4 text-sm font-medium hover:bg-zinc-50"
            >
              Cancel
            </button>
          )}

          <button
            onClick={() => file && onSubmit(file, language)}
            disabled={!file || busy}
            className="inline-flex h-10 items-center gap-2 rounded-lg bg-zinc-900 px-5 text-sm font-medium text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-300"
          >
            {busy && (
              <Loader2
                className="size-4 animate-spin"
                aria-hidden
              />
            )}

            {busy ? "Uploading…" : "Analyze"}
          </button>
        </div>
      </div>
    </div>
  );
}
