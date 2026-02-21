"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import Image from "next/image";
import {
  Send, Search, Zap, Database, BrainCircuit,
  Shield, Activity, Wifi, ChevronRight, LogOut,
  Sparkles, Globe, Layers, Clock,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// ---- types ----
type Message = {
  role: "user" | "agent";
  content: string;
  status?: string;
  progress?: number;
  timestamp?: string;
};

// ---- auth helpers ----
function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("ixchio_token");
}
function setToken(t: string) { localStorage.setItem("ixchio_token", t); }
function clearToken() { localStorage.removeItem("ixchio_token"); }

function getTime() {
  return new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
}

// ---- progress ring ----
function ProgressRing({ progress, size = 40, stroke = 3 }: { progress: number; size?: number; stroke?: number }) {
  const radius = (size - stroke) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;

  return (
    <svg width={size} height={size} className="rotate-[-90deg]">
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none"
        stroke="rgba(124,58,237,0.15)" strokeWidth={stroke} />
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none"
        stroke="url(#progress-gradient)" strokeWidth={stroke}
        strokeDasharray={circumference} strokeDashoffset={offset}
        strokeLinecap="round" className="transition-all duration-500" />
      <defs>
        <linearGradient id="progress-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#b5179e" />
          <stop offset="100%" stopColor="#4cc9f0" />
        </linearGradient>
      </defs>
    </svg>
  );
}


// ---- auth modal ----
function AuthModal({ onAuth }: { onAuth: () => void }) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setError("");
    setLoading(true);
    try {
      const endpoint = mode === "login" ? "/auth/login" : "/auth/signup";
      const body: Record<string, string> = { email, password };
      if (mode === "signup") body.name = name;

      const res = await fetch(`${API}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Auth failed");
      setToken(data.access_token);
      onAuth();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center noise-bg">
      {/* CRT effects */}
      <div className="crt-overlay" />
      <div className="crt-scanline-moving" />

      {/* Ambient glow orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#b5179e] rounded-full blur-[200px] opacity-10 animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-[#7c3aed] rounded-full blur-[180px] opacity-10" />

      <motion.div
        initial={{ opacity: 0, y: 30, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="glass-card p-10 w-full max-w-md space-y-6 relative z-10"
      >
        {/* Logo */}
        <div className="flex flex-col items-center gap-4 mb-2">
          <div className="relative">
            <Image src="/logo.png" alt="ixchio" width={56} height={56}
              className="pixelated relative z-10" style={{ imageRendering: "pixelated" }} />
            <div className="absolute inset-0 bg-[#b5179e] rounded-full blur-xl opacity-30 scale-150" />
          </div>
          <h1 className="font-pixel text-3xl tracking-[0.3em] glitch-text neon-text-purple" data-text="IXCHIO">
            IXCHIO
          </h1>
        </div>

        {/* mode tabs */}
        <div className="flex gap-1 bg-black/40 rounded-lg p-1">
          {(["login", "signup"] as const).map((m) => (
            <button key={m} onClick={() => setMode(m)}
              className={`flex-1 py-2.5 text-xs font-pixel tracking-[0.2em] uppercase rounded-md transition-all duration-300 ${mode === m
                ? "bg-gradient-to-r from-[#b5179e]/20 to-[#7c3aed]/20 text-white border border-[#b5179e]/30"
                : "text-zinc-500 hover:text-zinc-300"
                }`}
            >
              {m === "login" ? "ACCESS" : "REGISTER"}
            </button>
          ))}
        </div>

        {/* inputs */}
        <div className="space-y-3">
          <AnimatePresence>
            {mode === "signup" && (
              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}>
                <input type="text" placeholder="HANDLE" value={name}
                  onChange={e => setName(e.target.value)}
                  className="w-full bg-black/60 border border-zinc-800 focus:border-[#b5179e]/60 text-white font-mono text-sm px-4 py-3.5 rounded-lg outline-none transition-all focus:shadow-[0_0_15px_rgba(181,23,158,0.15)]" />
              </motion.div>
            )}
          </AnimatePresence>
          <input type="email" placeholder="EMAIL" value={email}
            onChange={e => setEmail(e.target.value)}
            className="w-full bg-black/60 border border-zinc-800 focus:border-[#b5179e]/60 text-white font-mono text-sm px-4 py-3.5 rounded-lg outline-none transition-all focus:shadow-[0_0_15px_rgba(181,23,158,0.15)]" />
          <input type="password" placeholder="PASSWORD" value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === "Enter" && submit()}
            className="w-full bg-black/60 border border-zinc-800 focus:border-[#b5179e]/60 text-white font-mono text-sm px-4 py-3.5 rounded-lg outline-none transition-all focus:shadow-[0_0_15px_rgba(181,23,158,0.15)]" />
        </div>

        {error && (
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="text-[#f72585] font-mono text-xs bg-[#f72585]/5 border border-[#f72585]/20 rounded-lg px-3 py-2">
            ⚠ {error}
          </motion.p>
        )}

        <button onClick={submit} disabled={loading}
          className="cyber-button cyber-button-primary w-full py-3.5 text-sm tracking-[0.2em] rounded-lg disabled:opacity-40">
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
              AUTHENTICATING...
            </span>
          ) : (
            mode === "login" ? "ACCESS SYSTEM" : "INITIALIZE ACCOUNT"
          )}
        </button>

        <p className="text-center text-zinc-600 text-[10px] font-pixel tracking-[0.15em] uppercase">
          {mode === "login" ? "No credentials?" : "Already initialized?"}{" "}
          <button onClick={() => setMode(mode === "login" ? "signup" : "login")}
            className="text-[#4cc9f0] hover:text-white transition-colors">
            {mode === "login" ? "Register" : "Access"}
          </button>
        </p>

        {/* decorative corner markers */}
        <div className="absolute top-3 left-3 w-3 h-3 border-l border-t border-[#b5179e]/30" />
        <div className="absolute top-3 right-3 w-3 h-3 border-r border-t border-[#b5179e]/30" />
        <div className="absolute bottom-3 left-3 w-3 h-3 border-l border-b border-[#b5179e]/30" />
        <div className="absolute bottom-3 right-3 w-3 h-3 border-r border-b border-[#b5179e]/30" />
      </motion.div>
    </div>
  );
}


// ---- main page ----
export default function Home() {
  const [authed, setAuthed] = useState(false);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { setAuthed(!!getToken()); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  // --- WS/poll data handler ---
  const handleWSData = useCallback((data: Record<string, unknown>) => {
    setMessages(prev => {
      const updated = [...prev];
      const last = updated[updated.length - 1];
      if (!last || last.role !== "agent") return updated;

      if (data.status === "completed") {
        last.content = data.report as string;
        last.status = "completed";
        last.progress = 100;
        setIsSearching(false);
      } else if (data.status === "failed") {
        last.content = `⚠️ ${data.error || "Pipeline failure"}`;
        last.status = "error";
        setIsSearching(false);
      } else {
        last.status = data.current_step as string;
        last.progress = data.progress as number;
        last.content = `${data.current_step || "Processing"}`;
      }
      return updated;
    });
  }, []);

  // HTTP polling fallback
  const pollTask = useCallback((taskId: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API}/api/v1/research/${taskId}`);
        if (!res.ok) { clearInterval(interval); setIsSearching(false); return; }
        const data = await res.json();
        handleWSData(data);
        if (data.status === "completed" || data.status === "failed") {
          clearInterval(interval);
        }
      } catch {
        clearInterval(interval);
        setIsSearching(false);
      }
    }, 2000);
  }, [handleWSData]);

  const connectWS = useCallback((taskId: string) => {
    const wsProto = API.startsWith("https") ? "wss" : "ws";
    const wsHost = API.replace(/^https?:\/\//, "");
    const ws = new WebSocket(`${wsProto}://${wsHost}/ws/research/${taskId}`);
    ws.onmessage = (ev) => handleWSData(JSON.parse(ev.data));
    ws.onerror = () => { ws.close(); pollTask(taskId); };
  }, [handleWSData, pollTask]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isSearching) return;

    const q = query.trim();
    setQuery("");
    setMessages(prev => [
      ...prev,
      { role: "user", content: q, timestamp: getTime() },
      { role: "agent", content: "Initializing research pipeline...", status: "starting", progress: 0, timestamp: getTime() },
    ]);
    setIsSearching(true);

    try {
      const token = getToken();
      const res = await fetch(`${API}/api/v1/research`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ query: q, depth: "medium", max_sources: 10, mode: "standard" }),
      });

      if (res.status === 401) {
        clearToken();
        setAuthed(false);
        setIsSearching(false);
        return;
      }

      const data = await res.json();
      if (data.task_id) connectWS(data.task_id);
    } catch {
      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last) {
          last.content = "Could not reach backend. Is the server running?";
          last.status = "error";
        }
        return updated;
      });
      setIsSearching(false);
    }
  };

  if (!authed) return <AuthModal onAuth={() => setAuthed(true)} />;

  return (
    <div className="flex h-screen overflow-hidden bg-[#030305] text-white selection:bg-[#b5179e]/30 noise-bg">
      {/* CRT effects */}
      <div className="crt-overlay" />
      <div className="crt-scanline-moving" />

      {/* ===== SIDEBAR ===== */}
      <motion.div
        initial={false}
        animate={{ width: sidebarOpen ? 260 : 0, opacity: sidebarOpen ? 1 : 0 }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        className="hidden md:flex flex-col glass-panel border-r border-[#b5179e]/10 overflow-hidden z-20 flex-shrink-0"
      >
        {/* Logo */}
        <div className="p-5 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Image src="/logo.png" alt="ixchio" width={36} height={36}
                className="pixelated relative z-10" style={{ imageRendering: "pixelated" }} />
              <div className="absolute inset-0 bg-[#b5179e] rounded-full blur-lg opacity-30 scale-150" />
            </div>
            <div>
              <h1 className="font-pixel text-lg tracking-[0.25em] neon-text-purple leading-none">IXCHIO</h1>
              <span className="text-[9px] font-pixel text-zinc-500 tracking-[0.2em]">DEEP RESEARCH v3.0</span>
            </div>
          </div>
        </div>

        {/* Nav */}
        <div className="flex-1 overflow-y-auto p-4 space-y-1">
          <div className="text-[9px] text-zinc-600 font-pixel uppercase tracking-[0.3em] mb-4 px-2">MODULES</div>

          {[
            { icon: <Sparkles size={15} />, label: "Deep Research", active: true, color: "#f72585" },
            { icon: <Globe size={15} />, label: "Web Search", active: false, color: "#4cc9f0" },
            { icon: <Layers size={15} />, label: "Knowledge Base", active: false, color: "#7c3aed" },
            { icon: <Clock size={15} />, label: "History", active: false, color: "#06d6a0" },
          ].map((item) => (
            <button key={item.label}
              className={`flex w-full items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-mono transition-all duration-200 group ${item.active
                ? "bg-gradient-to-r from-[rgba(181,23,158,0.1)] to-transparent border border-[#b5179e]/20 text-white"
                : "text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.02]"
                }`}
            >
              <span style={{ color: item.active ? item.color : undefined }} className="transition-colors group-hover:text-white">
                {item.icon}
              </span>
              <span className="tracking-wider">{item.label}</span>
              {item.active && <ChevronRight size={12} className="ml-auto text-[#b5179e]/50" />}
            </button>
          ))}
        </div>

        {/* System status */}
        <div className="p-4 border-t border-white/5 space-y-3">
          <div className="text-[9px] text-zinc-600 font-pixel uppercase tracking-[0.3em] px-1">SYSTEM</div>
          <div className="flex flex-wrap gap-1.5">
            <span className="status-badge status-badge-live"><span className="dot-live" /> VDB</span>
            <span className="status-badge status-badge-active"><span className="dot-active" /> STORM</span>
          </div>

          <button onClick={() => { clearToken(); setAuthed(false); }}
            className="flex items-center gap-2 text-zinc-600 hover:text-[#f72585] transition-colors text-[10px] font-pixel tracking-[0.15em] uppercase w-full px-1 py-1">
            <LogOut size={12} /> DISCONNECT
          </button>
        </div>
      </motion.div>

      {/* ===== MAIN AREA ===== */}
      <div className="flex-1 flex flex-col relative min-w-0">
        {/* Ambient glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-[#b5179e] rounded-full blur-[200px] opacity-[0.03] pointer-events-none" />
        <div className="absolute bottom-0 right-0 w-[400px] h-[400px] bg-[#7c3aed] rounded-full blur-[200px] opacity-[0.03] pointer-events-none" />

        {/* Header */}
        <header className="h-12 border-b border-white/5 flex items-center justify-between px-4 md:px-6 glass-panel z-30 flex-shrink-0">
          <div className="flex items-center gap-4">
            {/* Mobile toggle + sidebar toggle */}
            <button onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-zinc-500 hover:text-white transition-colors p-1 md:block hidden">
              <Activity size={14} />
            </button>
            {/* Mobile logo */}
            <div className="flex md:hidden items-center gap-2">
              <Image src="/logo.png" alt="ixchio" width={24} height={24} style={{ imageRendering: "pixelated" }} />
              <span className="font-pixel text-sm neon-text-purple tracking-[0.2em]">IXCHIO</span>
            </div>

            <div className="hidden sm:flex items-center gap-3">
              <span className="status-badge status-badge-live"><span className="dot-live" /><Database size={10} /> VDB_LIVE</span>
              <span className="status-badge status-badge-active"><span className="dot-active" /><BrainCircuit size={10} /> STORM_ACTIVE</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {isSearching && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="flex items-center gap-2 text-[#f72585] font-pixel text-[10px] tracking-[0.2em] uppercase">
                <Zap size={12} className="animate-pulse" /> RESEARCHING
              </motion.div>
            )}
            <div className="flex items-center gap-2 text-zinc-600 text-[10px] font-pixel tracking-wider">
              <Shield size={11} /> <Wifi size={11} />
            </div>
          </div>
        </header>

        {/* ===== CHAT AREA ===== */}
        <div className="flex-1 overflow-y-auto px-4 py-8 md:px-8 lg:px-16 z-10 pb-44">
          {messages.length === 0 ? (
            /* --- EMPTY STATE / HERO --- */
            <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="text-center space-y-8"
              >
                {/* Big logo with glow */}
                <div className="relative inline-block float-anim">
                  <Image src="/logo.png" alt="ixchio" width={80} height={80}
                    className="relative z-10" style={{ imageRendering: "pixelated" }} />
                  <div className="absolute inset-0 bg-[#b5179e] rounded-full blur-3xl opacity-20 scale-[2]" />
                  <div className="absolute inset-0 bg-[#7c3aed] rounded-full blur-2xl opacity-15 scale-[1.5]" />
                </div>

                <div>
                  <h2 className="font-pixel text-5xl md:text-6xl neon-text-purple tracking-[0.3em] mb-3 glitch-text" data-text="IXCHIO">
                    IXCHIO
                  </h2>
                  <p className="text-zinc-500 font-pixel text-sm tracking-[0.25em] uppercase">
                    Deep Research Engine
                  </p>
                </div>

                {/* Terminal-style info block */}
                <div className="glass-card p-6 text-left max-w-lg mx-auto">
                  <div className="font-mono text-xs leading-loose text-zinc-400 space-y-1">
                    <p><span className="text-[#b5179e]">$</span> ixchio <span className="text-zinc-600">--version</span> <span className="text-[#06d6a0]">3.0.0</span></p>
                    <p><span className="text-[#b5179e]">$</span> engines <span className="text-zinc-600">--list</span></p>
                    <p className="pl-4 text-[#4cc9f0]">→ STORM Multi-Perspective Analysis</p>
                    <p className="pl-4 text-[#4cc9f0]">→ Adaptive Search (Tavily + Jina)</p>
                    <p className="pl-4 text-[#4cc9f0]">→ Reflection Loop Synthesis</p>
                    <p><span className="text-[#b5179e]">$</span> status <span className="text-[#06d6a0]">READY</span> <span className="typing-cursor" /></p>
                  </div>
                </div>

                <p className="text-zinc-600 text-[10px] font-pixel tracking-[0.3em] uppercase">
                  Enter a research query below to begin
                </p>
              </motion.div>
            </div>
          ) : (
            /* --- MESSAGES --- */
            <div className="max-w-4xl mx-auto space-y-6">
              <AnimatePresence>
                {messages.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: 0.05 }}
                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {msg.role === "user" ? (
                      /* --- USER MESSAGE --- */
                      <div className="max-w-[85%] glass-card p-5 relative group border-[#4cc9f0]/20 hover:border-[#4cc9f0]/40">
                        <div className="flex items-center gap-2 mb-2">
                          <Search size={12} className="text-[#4cc9f0]" />
                          <span className="font-pixel text-[10px] text-[#4cc9f0]/60 tracking-[0.2em] uppercase">Query</span>
                          <span className="ml-auto text-[9px] text-zinc-700 font-mono">{msg.timestamp}</span>
                        </div>
                        <p className="text-zinc-200 font-mono text-sm leading-relaxed">{msg.content}</p>
                        {/* corner accent */}
                        <div className="absolute top-0 right-0 w-5 h-[2px] bg-gradient-to-l from-[#4cc9f0]/60 to-transparent" />
                        <div className="absolute top-0 right-0 w-[2px] h-5 bg-gradient-to-b from-[#4cc9f0]/60 to-transparent" />
                      </div>
                    ) : msg.status !== "completed" && msg.status !== "error" ? (
                      /* --- AGENT LOADING --- */
                      <div className="w-full max-w-[95%] glass-card p-6 border-[#b5179e]/20">
                        <div className="flex items-center gap-4">
                          <ProgressRing progress={msg.progress || 0} />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1.5">
                              <Zap size={12} className="text-[#f72585] animate-pulse" />
                              <span className="font-pixel text-[10px] text-[#f72585] tracking-[0.2em] uppercase">Researching</span>
                            </div>
                            <p className="text-zinc-400 font-mono text-xs truncate">{msg.content}</p>
                            {/* progress bar */}
                            <div className="mt-3 h-[2px] bg-zinc-900 rounded-full overflow-hidden">
                              <motion.div
                                className="h-full bg-gradient-to-r from-[#b5179e] to-[#4cc9f0] rounded-full"
                                initial={{ width: 0 }}
                                animate={{ width: `${msg.progress || 0}%` }}
                                transition={{ duration: 0.5 }}
                              />
                            </div>
                          </div>
                          <span className="font-pixel text-lg text-[#b5179e]/80 tracking-widest shrink-0">
                            {msg.progress || 0}%
                          </span>
                        </div>
                      </div>
                    ) : msg.status === "error" ? (
                      /* --- ERROR --- */
                      <div className="w-full max-w-[95%] glass-card p-5 border-[#f72585]/30">
                        <div className="flex items-center gap-2 text-[#f72585]">
                          <span className="font-pixel text-xs tracking-[0.15em]">⚠ ERROR</span>
                        </div>
                        <p className="text-zinc-400 font-mono text-sm mt-2">{msg.content}</p>
                      </div>
                    ) : (
                      /* --- COMPLETED RESEARCH --- */
                      <div className="w-full max-w-[95%] glass-card overflow-hidden border-[#b5179e]/20">
                        {/* result header */}
                        <div className="px-6 py-3 border-b border-white/5 flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Sparkles size={13} className="text-[#b5179e]" />
                            <span className="font-pixel text-[10px] text-zinc-400 tracking-[0.2em] uppercase">Research Complete</span>
                          </div>
                          <span className="text-[9px] text-zinc-700 font-mono">{msg.timestamp}</span>
                        </div>
                        {/* content */}
                        <div className="p-6 md:p-8 prose prose-invert max-w-none research-prose text-sm leading-relaxed">
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        </div>
                        {/* result footer */}
                        <div className="px-6 py-3 border-t border-white/5 flex items-center gap-4 text-[9px] font-pixel text-zinc-600 tracking-[0.15em] uppercase">
                          <span className="flex items-center gap-1.5"><Database size={10} /> Sources Indexed</span>
                          <span className="flex items-center gap-1.5"><BrainCircuit size={10} /> STORM Analyzed</span>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
              <div ref={endRef} className="h-4" />
            </div>
          )}
        </div>

        {/* ===== INPUT BAR ===== */}
        <div className="absolute bottom-0 left-0 right-0 p-4 md:p-6 z-30">
          {/* Gradient fade */}
          <div className="absolute inset-0 bg-gradient-to-t from-[#030305] via-[#030305]/95 to-transparent pointer-events-none" />

          <div className="max-w-4xl mx-auto relative">
            <form onSubmit={submit}>
              <div className="gradient-input-wrapper">
                <div className="gradient-input-inner flex items-center">
                  <div className="pl-5 text-[#b5179e]/50">
                    <Search size={18} />
                  </div>
                  <input
                    type="text" value={query} onChange={e => setQuery(e.target.value)}
                    placeholder="What would you like to research?"
                    disabled={isSearching} spellCheck={false}
                    className="flex-1 bg-transparent text-white font-mono text-sm pl-4 pr-4 py-4 md:py-5 outline-none placeholder:text-zinc-600 disabled:opacity-50"
                  />
                  <button type="submit" disabled={!query.trim() || isSearching}
                    className="mr-3 p-2.5 rounded-lg bg-gradient-to-r from-[#b5179e] to-[#7c3aed] text-white disabled:opacity-30 hover:shadow-[0_0_15px_rgba(181,23,158,0.4)] transition-all disabled:hover:shadow-none">
                    <Send size={16} />
                  </button>
                </div>
              </div>
            </form>

            {/* Footer info */}
            <div className="flex items-center justify-center gap-4 mt-3 text-[9px] font-pixel text-zinc-700 tracking-[0.25em] uppercase">
              <span>IXCHIO v3.0</span>
              <span className="w-1 h-1 bg-zinc-800 rounded-full" />
              <button
                type="button"
                onClick={() => setQuery("DEMO")}
                className="text-[#4cc9f0] hover:text-[#f72585] transition-colors border-b border-transparent hover:border-[#f72585]"
                title="Load pre-computed demo report instantly"
              >
                LOAD DEMO
              </button>
              <span className="w-1 h-1 bg-zinc-800 rounded-full" />
              <span>Multi-Agent</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
