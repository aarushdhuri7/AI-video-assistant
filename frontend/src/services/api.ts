
import type { Analysis, AnalysisSummary, Language, Status } from "../types";

/**
 * The only place that knows the backend address.
 * Local development uses localhost; Render uses VITE_API_BASE_URL.
 */
export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL?.trim() ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const REQUEST_TIMEOUT_MS = 30_000;
const CHAT_TIMEOUT_MS = 5 * 60_000;
const UPLOAD_TIMEOUT_MS = 30 * 60_000;
const POLL_INTERVAL_MS = 2_000;
const POLL_GIVE_UP_MS = 3 * 60 * 60_000;
const MAX_POLL_FAILURES = 5;

export class ApiError extends Error {
  constructor(
    message: string,
    public status = 0,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

const NETWORK =
  "Can't reach the backend. Check your internet connection and backend URL, then try again.";

const STATUSES: Status[] = [
  "pending",
  "processing",
  "completed",
  "error",
];

function detailOf(body: unknown, fallback: string): string {
  if (typeof body !== "object" || body === null) {
    return fallback;
  }

  const detail = (body as { detail?: unknown }).detail;

  return typeof detail === "string" && detail.trim()
    ? detail
    : fallback;
}

function isAnalysis(value: unknown): value is Analysis {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const analysis = value as Partial<Analysis>;

  return (
    typeof analysis.id === "string" &&
    typeof analysis.status === "string" &&
    STATUSES.includes(analysis.status as Status) &&
    typeof analysis.filename === "string"
  );
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = REQUEST_TIMEOUT_MS,
  signal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();

  const abortFromParent = () => {
    controller.abort(signal?.reason);
  };

  if (signal?.aborted) {
    abortFromParent();
  } else {
    signal?.addEventListener("abort", abortFromParent, {
      once: true,
    });
  }

  const timer = setTimeout(() => {
    controller.abort("timeout");
  }, timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });

    let body: unknown = null;

    try {
      body = await response.json();
    } catch {
      // The response was not valid JSON.
    }

    if (!response.ok) {
      throw new ApiError(
        detailOf(body, `Request failed (${response.status}).`),
        response.status,
      );
    }

    if (body === null) {
      throw new ApiError("The server sent an invalid response.");
    }

    return body as T;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    if (signal?.aborted) {
      throw error;
    }

    if (controller.signal.aborted) {
      throw new ApiError("The request timed out. Please try again.");
    }

    throw new ApiError(NETWORK);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abortFromParent);
  }
}

export function listAnalyses(): Promise<AnalysisSummary[]> {
  return request<AnalysisSummary[]>("/api/analyses");
}

export async function getAnalysis(
  id: string,
  signal?: AbortSignal,
): Promise<Analysis> {
  const data = await request<unknown>(
    `/api/analyses/${encodeURIComponent(id)}`,
    {},
    REQUEST_TIMEOUT_MS,
    signal,
  );

  if (!isAnalysis(data)) {
    throw new ApiError("The server sent an invalid response.");
  }

  return data;
}

/**
 * Upload a file with real upload progress.
 * Fetch does not expose upload progress, so XMLHttpRequest is used here.
 */
export function uploadFile(
  file: File,
  language: Language,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<Analysis> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(
        signal.reason ??
          new DOMException("Upload cancelled", "AbortError"),
      );
      return;
    }

    const xhr = new XMLHttpRequest();

    xhr.open("POST", `${API_BASE_URL}/api/analyses/upload`);
    xhr.timeout = UPLOAD_TIMEOUT_MS;

    const abortUpload = () => xhr.abort();

    signal?.addEventListener("abort", abortUpload, {
      once: true,
    });

    const cleanup = () => {
      signal?.removeEventListener("abort", abortUpload);
    };

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(event.loaded / event.total);
      }
    };

    xhr.onload = () => {
      cleanup();

      let body: unknown = null;

      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // The response was not valid JSON.
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        if (isAnalysis(body)) {
          resolve(body);
        } else {
          reject(
            new ApiError("The server sent an invalid response."),
          );
        }

        return;
      }

      reject(
        new ApiError(
          detailOf(body, `Upload failed (${xhr.status}).`),
          xhr.status,
        ),
      );
    };

    xhr.onerror = () => {
      cleanup();
      reject(new ApiError(NETWORK));
    };

    xhr.ontimeout = () => {
      cleanup();
      reject(new ApiError("The upload timed out."));
    };

    xhr.onabort = () => {
      cleanup();
      reject(
        new DOMException("Upload cancelled", "AbortError"),
      );
    };

    const form = new FormData();
    form.append("file", file);
    form.append("language", language);

    xhr.send(form);
  });
}

export async function askQuestion(
  id: string,
  question: string,
): Promise<string> {
  const response = await request<{ answer?: unknown }>(
    `/api/analyses/${encodeURIComponent(id)}/chat`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    },
    CHAT_TIMEOUT_MS,
  );

  if (typeof response.answer !== "string") {
    throw new ApiError("The server sent an invalid response.");
  }

  return response.answer;
}

/**
 * Poll until the analysis completes or fails.
 * Tolerates temporary network failures and supports cancellation.
 */
export async function pollAnalysis(
  id: string,
  onUpdate: (analysis: Analysis) => void,
  signal: AbortSignal,
): Promise<Analysis> {
  const started = Date.now();
  let failures = 0;

  for (;;) {
    if (signal.aborted) {
      throw (
        signal.reason ??
        new DOMException("Polling cancelled", "AbortError")
      );
    }

    try {
      const analysis = await getAnalysis(id, signal);

      failures = 0;
      onUpdate(analysis);

      if (
        analysis.status === "completed" ||
        analysis.status === "error"
      ) {
        return analysis;
      }
    } catch (error) {
      if (signal.aborted) {
        throw error;
      }

      if (error instanceof ApiError && error.status === 404) {
        throw error;
      }

      failures += 1;

      if (failures >= MAX_POLL_FAILURES) {
        throw error;
      }
    }

    if (Date.now() - started > POLL_GIVE_UP_MS) {
      throw new ApiError(
        "Stopped waiting: the analysis is taking too long.",
      );
    }

    await new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        clearTimeout(timer);
        reject(
          signal.reason ??
            new DOMException("Polling cancelled", "AbortError"),
        );
      };

      const timer = setTimeout(() => {
        signal.removeEventListener("abort", onAbort);
        resolve();
      }, POLL_INTERVAL_MS);

      signal.addEventListener("abort", onAbort, {
        once: true,
      });

      if (signal.aborted) {
        onAbort();
      }
    });
  }
}
