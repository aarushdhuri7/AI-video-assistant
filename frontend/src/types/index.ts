
export type Language = "english" | "hindi" | "auto" | "hinglish";

export type Status = "pending" | "processing" | "completed" | "error";

export interface AnalysisSummary {
  id: string;
  status: Status;
  filename: string;
  language: Language;
  created_at: string;
  title: string | null;
}

export interface Analysis extends AnalysisSummary {
  stage: string | null;
  error: string | null;
  transcript: string | null;
  summary: string | null;
  action_items: string | null;
  key_decisions: string | null;
  open_questions: string | null;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  text: string;
  isError?: boolean;
}

export type View = "new" | "results" | "chat";
