import { useEffect, useState } from "react";
import axios from "axios";
import { getHealth, getHistory, getMe, getReady, getTrends, login, predict, register } from "./api";
import "./styles.css";
import type { HistoryItem, PredictResponse, ReadyResponse, TrendResponse, User } from "./types";

type Scenario = "distress" | "celebration" | "frustration" | "neutral" | "seeking_help" | "gratitude";
type AuthMode = "login" | "register";

type Preset = {
  label: string;
  scenario: Scenario;
  text: string;
};

const SESSION_KEY = "cameo_session_token";
const MAX_TEXT_CHARS = 1200;

const PRESETS: Preset[] = [
  { label: "Distress", scenario: "distress", text: "I feel very low and alone today." },
  { label: "Celebration", scenario: "celebration", text: "I got selected and I am so happy." },
  { label: "Frustration", scenario: "frustration", text: "I am frustrated because nothing works." },
  { label: "Neutral", scenario: "neutral", text: "Today was normal and routine." },
  { label: "Help", scenario: "seeking_help", text: "I feel anxious and I need guidance on what to do next." },
  { label: "Gratitude", scenario: "gratitude", text: "Thank you for supporting me, I feel hopeful now." },
];

const COLOR_BY_SCENARIO: Record<Scenario, string> = {
  distress: "#5f91d2",
  celebration: "#ffd25b",
  frustration: "#de6767",
  neutral: "#a7b0b8",
  seeking_help: "#8ab0d4",
  gratitude: "#80c091",
};

async function makeScenarioImage(scenario: Scenario): Promise<File> {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not create demo image.");

  ctx.fillStyle = COLOR_BY_SCENARIO[scenario];
  ctx.fillRect(0, 0, 224, 224);
  ctx.strokeStyle = "#1c2430";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(112, 112, 58, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "#1c2430";
  ctx.beginPath();
  ctx.arc(94, 100, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(130, 100, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#1c2430";
  ctx.lineWidth = 3;
  ctx.beginPath();
  if (scenario === "celebration" || scenario === "gratitude") ctx.arc(112, 122, 24, 0.2, Math.PI - 0.2);
  else if (scenario === "distress") ctx.arc(112, 146, 20, Math.PI + 0.2, Math.PI * 2 - 0.2);
  else if (scenario === "frustration") {
    ctx.moveTo(90, 136);
    ctx.lineTo(134, 122);
  } else if (scenario === "seeking_help") {
    ctx.moveTo(92, 138);
    ctx.lineTo(132, 132);
  } else {
    ctx.moveTo(94, 134);
    ctx.lineTo(130, 134);
  }
  ctx.stroke();

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) throw new Error("Could not encode demo image.");
  return new File([blob], `${scenario}-preset.png`, { type: "image/png" });
}

function toneClass(intent: string): string {
  if (intent === "distress") return "tone-distress";
  if (intent === "celebration") return "tone-celebration";
  if (intent === "frustration") return "tone-frustration";
  if (intent === "seeking_help") return "tone-help";
  if (intent === "gratitude") return "tone-gratitude";
  return "tone-neutral";
}

export default function App() {
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [activePreset, setActivePreset] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [copied, setCopied] = useState(false);
  const [serviceVersion, setServiceVersion] = useState("");
  const [serviceReady, setServiceReady] = useState<ReadyResponse | null>(null);
  const [serviceState, setServiceState] = useState<"checking" | "ready" | "degraded" | "offline">("checking");
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [sessionToken, setSessionToken] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [trends, setTrends] = useState<TrendResponse | null>(null);
  const remainingChars = MAX_TEXT_CHARS - text.length;
  const canRun = Boolean(file && text.trim() && text.length <= MAX_TEXT_CHARS);

  const loadUserData = async (token: string) => {
    const [me, items, trendData] = await Promise.all([getMe(token), getHistory(token), getTrends(token)]);
    setUser(me);
    setHistory(items);
    setTrends(trendData);
  };

  useEffect(() => {
    const token = window.localStorage.getItem(SESSION_KEY) ?? "";
    if (token) setSessionToken(token);
  }, []);

  useEffect(() => {
    if (sessionToken) window.localStorage.setItem(SESSION_KEY, sessionToken);
    else window.localStorage.removeItem(SESSION_KEY);
  }, [sessionToken]);

  useEffect(() => {
    let active = true;
    const loadStatus = async () => {
      try {
        const health = await getHealth();
        if (active) setServiceVersion(health.version);
        try {
          const ready = await getReady();
          if (!active) return;
          setServiceReady(ready);
          setServiceState("ready");
        } catch (e: unknown) {
          if (!active) return;
          if (axios.isAxiosError(e) && e.response?.status === 503) {
            setServiceReady((e.response.data?.detail as ReadyResponse | undefined) ?? null);
            setServiceState("degraded");
          } else {
            setServiceState("offline");
          }
        }
      } catch {
        if (active) {
          setServiceReady(null);
          setServiceState("offline");
        }
      }
    };
    loadStatus();
    const intervalId = window.setInterval(loadStatus, 15000);
    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    if (!sessionToken) {
      setUser(null);
      setHistory([]);
      setTrends(null);
      return;
    }
    loadUserData(sessionToken).catch(() => {
      setSessionToken("");
      setUser(null);
      setHistory([]);
      setTrends(null);
    });
  }, [sessionToken]);

  useEffect(() => {
    if (!file) {
      setImageUrl("");
      return;
    }
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const handlePreset = async (preset: Preset) => {
    setText(preset.text);
    setActivePreset(preset.label);
    try {
      setFile(await makeScenarioImage(preset.scenario));
      setError(null);
      setNotice(null);
    } catch {
      setError("Could not create preset image. Please upload an image manually.");
    }
  };

  const handleAuth = async () => {
    if (!authUsername.trim() || !authPassword.trim()) {
      setError("Enter a username and password to continue.");
      setNotice(null);
      return;
    }
    setAuthLoading(true);
    setError(null);
    setNotice(null);
    try {
      const response =
        authMode === "login" ? await login(authUsername.trim(), authPassword) : await register(authUsername.trim(), authPassword);
      setSessionToken(response.token);
      setUser(response.user);
      setAuthPassword("");
      setNotice(authMode === "login" ? `Logged in successfully as ${response.user.username}.` : `Registered successfully as ${response.user.username}.`);
    } catch (e: unknown) {
      if (axios.isAxiosError(e)) {
        setError((e.response?.data as { detail?: string } | undefined)?.detail ?? e.message);
      } else {
        setError("Authentication failed.");
      }
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    setSessionToken("");
    setUser(null);
    setHistory([]);
    setTrends(null);
    setNotice("Signed out.");
  };

  const handleSubmit = async () => {
    if (!file || !text.trim()) {
      setError("Please choose an image and enter text (or use a preset).");
      return;
    }
    if (text.length > MAX_TEXT_CHARS) {
      setError(`Caption is too long. Please keep it under ${MAX_TEXT_CHARS} characters.`);
      return;
    }
    setError(null);
    setNotice(null);
    setLoading(true);
    try {
      const res = await predict(text, file, sessionToken || undefined);
      setResult(res);
      if (sessionToken) {
        await loadUserData(sessionToken);
        setNotice("Prediction completed and saved to your history.");
      }
    } catch (e: unknown) {
      if (axios.isAxiosError(e)) {
        const rawDetail = e.response?.data?.detail;
        setError(typeof rawDetail === "string" ? rawDetail : e.message ?? "Request failed");
      } else {
        setError("Request failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!result?.response.text) return;
    try {
      await navigator.clipboard.writeText(result.response.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setError("Could not copy the reply. Please copy it manually.");
    }
  };

  if (!user) {
    return (
      <div className="shell shell-auth">
        <section className="hero card-rise auth-hero">
          <div className="auth-hero-copy">
            <p className="kicker">CAMEO Access</p>
            <h1>Sign in to CAMEO Studio</h1>
            <p className="hero-copy">
              Register or login first. The prediction dashboard, saved history, and trend analysis unlock after successful authentication.
            </p>
            <div className="status-row">
              <span className={`status-badge status-${serviceState}`}>
                {serviceState === "checking" && "Checking service"}
                {serviceState === "ready" && "Service ready"}
                {serviceState === "degraded" && "Service degraded"}
                {serviceState === "offline" && "Backend offline"}
              </span>
              {serviceVersion && <span className="status-detail">API v{serviceVersion}</span>}
              {serviceReady && <span className="status-detail">Device: {serviceReady.device}</span>}
            </div>
          </div>
          <div className="auth-side-card">
            <p className="auth-side-kicker">Workspace</p>
            <h3>Private session analytics</h3>
            <p>Every signed-in run stores emotion, intent, confidence, and response history for a cleaner live demo.</p>
          </div>
        </section>

        <section className="panel card-rise auth-panel">
          <div className="auth-panel-head">
            <p className="auth-panel-kicker">{authMode === "login" ? "Welcome back" : "Create your workspace"}</p>
            <h2>{authMode === "login" ? "Login to continue" : "Register a new account"}</h2>
          </div>
          <div className="pill-row auth-tabs">
            <button className={`ghost small ${authMode === "login" ? "active-tab" : ""}`} onClick={() => setAuthMode("login")}>
              Login
            </button>
            <button className={`ghost small ${authMode === "register" ? "active-tab" : ""}`} onClick={() => setAuthMode("register")}>
              Register
            </button>
          </div>
          <div className="auth-form">
            <input value={authUsername} onChange={(e) => setAuthUsername(e.target.value)} placeholder="Username" />
            <input value={authPassword} onChange={(e) => setAuthPassword(e.target.value)} placeholder="Password" type="password" />
            <button onClick={handleAuth} disabled={authLoading}>
              {authLoading ? "Please wait..." : authMode === "login" ? "Login" : "Create account"}
            </button>
            {notice && <p className="success-text">{notice}</p>}
            {error && <p className="error-text">{error}</p>}
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="shell">
      <header className="hero card-rise dashboard-hero">
        <div className="dashboard-hero-copy">
          <p className="kicker">CAMEO Dashboard</p>
          <h1>Welcome, {user.username}</h1>
          <p className="hero-copy">Upload an image, add a caption, run the model, and review your saved history and trend analysis.</p>
          <div className="status-row">
            <span className={`status-badge status-${serviceState}`}>
              {serviceState === "checking" && "Checking service"}
              {serviceState === "ready" && "Service ready"}
              {serviceState === "degraded" && "Service degraded"}
              {serviceState === "offline" && "Backend offline"}
            </span>
            {serviceVersion && <span className="status-detail">API v{serviceVersion}</span>}
            {serviceReady && <span className="status-detail">Weights: {serviceReady.weights_loaded ? "Loaded" : "Missing"}</span>}
          </div>
        </div>
        <div className="hero-actions">
          <button className="ghost" onClick={handleLogout}>Sign Out</button>
        </div>
      </header>

      <section className="card-rise panel">
        <div className="history-head">
          <h2>Account Snapshot</h2>
          <span className="status-detail">{history.length} stored checks</span>
        </div>
        <div className="ops-grid snapshot-grid">
          <div className="note stat-card">
            <strong>User</strong>
            <p>{user.username}</p>
          </div>
          <div className="note stat-card">
            <strong>Total Checks</strong>
            <p>{trends?.total_checks ?? 0}</p>
          </div>
          <div className="note stat-card">
            <strong>Top Emotion</strong>
            <p>{trends?.top_emotion ?? "No data yet"}</p>
          </div>
          <div className="note stat-card">
            <strong>Top Intent</strong>
            <p>{trends?.top_intent ?? "No data yet"}</p>
          </div>
          <div className="note stat-card">
            <strong>Latest Emotion</strong>
            <p>{history.length > 0 ? history[0].emotion : result?.emotion ?? "No data yet"}</p>
          </div>
          <div className="note stat-card">
            <strong>Latest Intent</strong>
            <p>{history.length > 0 ? history[0].intent : result?.intent ?? "No data yet"}</p>
          </div>
        </div>
      </section>

      <section className="layout">
        <div className="card-rise panel">
          <h2>New Check</h2>
          <label>Optional starter</label>
          <div className="preset-grid">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                className={`preset-chip chip-${preset.scenario} ${activePreset === preset.label ? "active" : ""}`}
                onClick={() => handlePreset(preset)}
              >
                {preset.label}
              </button>
            ))}
          </div>
          <label>Caption</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type your caption..."
            maxLength={MAX_TEXT_CHARS + 100}
          />
          <p className={`help-text ${remainingChars < 0 ? "help-text-error" : ""}`}>
            {text.length} / {MAX_TEXT_CHARS} characters
          </p>
          <label>Image</label>
          <input type="file" accept="image/png,image/jpeg" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          <div className="image-preview">{imageUrl ? <img src={imageUrl} alt="Preview" /> : <p>No image selected</p>}</div>
          <div className="actions">
            <button onClick={handleSubmit} disabled={loading || !canRun}>
              {loading ? "Analyzing..." : "Run CAMEO"}
            </button>
            <button
              className="ghost"
              onClick={() => {
                setText("");
                setFile(null);
                setActivePreset("");
                setResult(null);
                setError(null);
                setNotice(null);
              }}
            >
              Reset
            </button>
          </div>
          {notice && <p className="success-text">{notice}</p>}
          {error && <p className="error-text">{error}</p>}
        </div>

        <div className="card-rise panel">
          <h2>Latest Result</h2>
          {!result && <p className="placeholder">Run a prediction to see the latest output.</p>}
          {result && (
            <>
              <div className="pill-row">
                <span className="pill">Emotion: {result.emotion}</span>
                <span className="pill">Intent: {result.intent}</span>
                <span className="pill">Confidence: {(result.confidence * 100).toFixed(1)}%</span>
                <span className="pill">Intensity: {result.intensity.toFixed(2)}</span>
              </div>
              <div className={`response-box ${toneClass(result.intent)}`}>
                <h3>Supportive Reply</h3>
                <p>{result.response.text}</p>
                <div className="inline-actions">
                  <button className="ghost" onClick={handleCopy}>
                    {copied ? "Copied" : "Copy Reply"}
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </section>

      <section className="card-rise panel">
        <div className="history-head">
          <h2>User History</h2>
          <span className="status-detail">Saved predictions</span>
        </div>
        {history.length === 0 && <p className="placeholder">No saved checks yet. Run CAMEO once to start history tracking.</p>}
        {history.length > 0 && (
          <div className="history-grid">
            {history.map((item) => (
              <article key={item.id} className="history-item">
                <p className="history-text">{item.text}</p>
                <p className="history-meta">
                  Emotion: {item.emotion} | Intent: {item.intent} | Confidence: {(item.confidence * 100).toFixed(1)}% | {new Date(item.created_at).toLocaleString()}
                </p>
                <p className="history-response">{item.response_text}</p>
              </article>
            ))}
          </div>
        )}
      </section>

      <section className="card-rise panel">
        <div className="history-head">
          <h2>Trend Analysis</h2>
          <span className="status-detail">From your recent checks</span>
        </div>
        <div className="notes-grid">
          <div className="note">
            <strong>Most Frequent Emotion</strong>
            <p>{trends?.top_emotion ?? "No data yet"}</p>
          </div>
          <div className="note">
            <strong>Most Frequent Intent</strong>
            <p>{trends?.top_intent ?? "No data yet"}</p>
          </div>
          <div className="note">
            <strong>Average Intensity</strong>
            <p>{trends ? trends.average_intensity.toFixed(2) : "0.00"}</p>
          </div>
          <div className="note">
            <strong>Average Confidence</strong>
            <p>{trends ? `${(trends.average_confidence * 100).toFixed(1)}%` : "0.0%"}</p>
          </div>
          <div className="note">
            <strong>Recent Emotions</strong>
            <p>{trends?.recent_emotions.length ? trends.recent_emotions.join(", ") : "No data yet"}</p>
          </div>
          <div className="note">
            <strong>Recent Intents</strong>
            <p>{trends?.recent_intents.length ? trends.recent_intents.join(", ") : "No data yet"}</p>
          </div>
        </div>
      </section>
    </div>
  );
}
