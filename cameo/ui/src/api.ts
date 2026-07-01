import axios from "axios";
import type { AuthResponse, HealthResponse, HistoryItem, PredictResponse, ReadyResponse, TrendResponse, User } from "./types";

const rawBaseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
const baseURL = rawBaseUrl && rawBaseUrl.trim() ? rawBaseUrl : "";

const client = axios.create({
  baseURL,
  timeout: 30000,
});

function authHeaders(token?: string) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function predict(text: string, image: File, token?: string): Promise<PredictResponse> {
  const form = new FormData();
  form.append("text", text);
  form.append("image", image);
  const res = await client.post<PredictResponse>("/predict", form, {
    headers: { "Content-Type": "multipart/form-data", ...authHeaders(token) },
  });
  return res.data;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await client.get<HealthResponse>("/health");
  return res.data;
}

export async function getReady(): Promise<ReadyResponse> {
  const res = await client.get<ReadyResponse>("/ready");
  return res.data;
}

export async function register(username: string, password: string): Promise<AuthResponse> {
  const res = await client.post<AuthResponse>("/auth/register", { username, password });
  return res.data;
}

export async function login(username: string, password: string): Promise<AuthResponse> {
  const res = await client.post<AuthResponse>("/auth/login", { username, password });
  return res.data;
}

export async function getMe(token: string): Promise<User> {
  const res = await client.get<User>("/auth/me", { headers: authHeaders(token) });
  return res.data;
}

export async function getHistory(token: string): Promise<HistoryItem[]> {
  const res = await client.get<HistoryItem[]>("/history", { headers: authHeaders(token) });
  return res.data;
}

export async function getTrends(token: string): Promise<TrendResponse> {
  const res = await client.get<TrendResponse>("/trends", { headers: authHeaders(token) });
  return res.data;
}
