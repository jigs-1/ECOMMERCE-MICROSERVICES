export type PredictResponse = {
  emotion: string;
  intensity: number;
  intent: string;
  confidence: number;
  attn_weights: Record<string, number>;
  gates: Record<string, number>;
  response: { text: string; mode: "rule" | "generative" };
};

export type User = {
  id: number;
  username: string;
  created_at: string;
};

export type AuthResponse = {
  token: string;
  user: User;
};

export type HistoryItem = {
  id: number;
  text: string;
  emotion: string;
  intent: string;
  intensity: number;
  confidence: number;
  response_text: string;
  created_at: string;
};

export type TrendResponse = {
  total_checks: number;
  top_emotion: string | null;
  top_intent: string | null;
  average_intensity: number;
  average_confidence: number;
  recent_emotions: string[];
  recent_intents: string[];
};

export type HealthResponse = {
  status: string;
  version: string;
};

export type ReadyResponse = {
  status: "ready" | "degraded";
  version: string;
  device: string;
  model_loaded: boolean;
  tokenizer_loaded: boolean;
  weights_present: boolean;
  weights_loaded: boolean;
  weights_path: string;
  frontend_built: boolean;
};
