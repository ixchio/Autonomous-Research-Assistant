"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import Image from "next/image";
import { Terminal, Send, Search, Settings, Zap, Database, BrainCircuit, LogIn, UserPlus } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const API = "http://127.0.0.1:8000";

// ---- types ----
type Message = {
  role: "user" | "agent";
  content: string;
  status?: string;
  progress?: number;
};

// ---- auth helpers ----
function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("ixchio_token");
}
function setToken(t: string) { localStorage.setItem("ixchio_token", t); }
function clearToken() { localStorage.removeItem("ixchio_token"); }

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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90">
      <div className="pixel-panel p-8 w-full max-w-md space-y-6">
        <h2 className="font-pixel text-2xl text-center neon-text-purple tracking-widest">
          {mode === "login" ? "LOGIN" : "SIGN_UP"}
        </h2>

        {mode === "signup" && (
          <input
            type="text" placeholder="Name" value={name}
            onChange={e => setName(e.target.value)}
            className="w-full bg-black border-2 border-zinc-700 focus:border-[#b5179e] text-white font-mono text-sm px-4 py-3 outline-none"
          />
        )}
        <input
          type="email" placeholder="Email" value={email}
          onChange={e => setEmail(e.target.value)}
          className="w-full bg-black border-2 border-zinc-700 focus:border-[#b5179e] text-white font-mono text-sm px-4 py-3 outline-none"
        />
        <input
          type="password" placeholder="Password" value={password}
          onChange={e => setPassword(e.target.value)}
          onKeyDown={e => e.key === "Enter" && submit()}
          className="w-full bg-black border-2 border-zinc-700 focus:border-[#b5179e] text-white font-mono text-sm px-4 py-3 outline-none"
        />

        {error && <p className="text-[#f72585] font-mono text-xs">{error}</p>}

        <button
          onClick={submit} disabled={loading}
          className="pixel-button w-full py-3 text-sm disabled:opacity-40"
        >
          {loading ? "..." : mode === "login" ? "ACCESS SYSTEM" : "CREATE ACCOUNT"}
        </button>

        <p className="text-center text-zinc-500 text-xs font-mono">
          {mode === "login" ? "No account?" : "Already in?"}{" "}
          <button
            onClick={() => setMode(mode === "login" ? "signup" : "login")}
            className="text-[#4cc9f0] hover:underline"
          >
            {mode === "login" ? "Sign up" : "Log in"}
          </button>
        </p>
      </div>
    </div>
  );
}


// ---- main page ----
export default function Home() {
  const [authed, setAuthed] = useState(false);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { setAuthed(!!getToken()); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const connectWS = (taskId: string) => {
    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/research/${taskId}`);

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      setMessages(prev => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (!last || last.role !== "agent") return updated;

        if (data.status === "completed") {
          last.content = data.report;
          last.status = "completed";
          last.progress = 100;
          setIsSearching(false);
        } else if (data.status === "failed") {
          last.content = `\u26a0\ufe0f ${data.error || "Pipeline failure"}`;
          last.status = "error";
          setIsSearching(false);
        } else {
          last.status = data.current_step;
          last.progress = data.progress;
          last.content = `\u23f3 ${data.current_step || "Processing"}... [${data.progress}%]`;
        }
        return updated;
      });
    };
    ws.onerror = () => setIsSearching(false);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isSearching) return;

    const q = query.trim();
    setQuery("");
    setMessages(prev => [
      ...prev,
      { role: "user", content: q },
      { role: "agent", content: "\u23f3 Initializing...", status: "starting", progress: 0 },
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
          last.content = "\u26a0\ufe0f Could not reach backend. Is Python server running?";
          last.status = "error";
        }
        return updated;
      });
      setIsSearching(false);
    }
  };

  if (!authed) return <AuthModal onAuth={() => setAuthed(true)} />;

  return (
    <div className="flex h-screen overflow-hidden bg-black text-white selection:bg-[#b5179e]/30">
      {/* sidebar */}
      <div className="hidden md:flex w-64 flex-col border-r-2 border-zinc-800 bg-[#050505] z-20">
        <div className="p-5 border-b-2 border-zinc-800">
          <h1 className="flex items-center gap-3 font-pixel text-xl neon-text-purple tracking-widest pl-2">
            <Image src="/logo.png" alt="ixchio logo" width={32} height={32} className="pixelated" />
            IXCHIO
          </h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          <div className="text-[10px] text-zinc-500 font-pixel uppercase tracking-[0.2em] mb-3">Engines</div>
          <button className="pixel-button flex w-full items-center gap-3 px-3 py-3 text-sm rounded-none">
            <Search size={16} /> Core Search
          </button>
          <button className="pixel-button flex w-full items-center gap-3 px-3 py-3 text-sm rounded-none !border-[#f72585] !text-[#f72585]">
            <Zap size={16} /> Deep Research
          </button>
        </div>

        <div className="p-4 border-t-2 border-zinc-800 flex justify-between items-center text-zinc-500 font-pixel text-[10px] uppercase tracking-widest">
          <span>v3.0</span>
          <button onClick={() => { clearToken(); setAuthed(false); }} className="hover:text-white transition-colors">
            LOGOUT
          </button>
        </div>
      </div>

      {/* main */}
      <div className="flex-1 flex flex-col relative">
        <div className="absolute inset-0 pointer-events-none opacity-10 z-50 bg-[linear-gradient(rgba(0,0,0,0)_50%,rgba(0,0,0,0.25)_50%)] bg-[length:100%_4px]" />

        <header className="h-14 border-b-2 border-zinc-800 flex items-center justify-between px-6 bg-[#030303]/90 backdrop-blur-md z-30">
          <div className="flex items-center gap-6 text-[10px] font-pixel text-zinc-500 uppercase tracking-[0.15em]">
            <span className="flex items-center gap-2 text-zinc-300">
              <Database size={12} className="text-[#4cc9f0]" /> VDB_<span className="text-[#4cc9f0]">LIVE</span>
            </span>
            <span className="flex items-center gap-2 text-zinc-300">
              <BrainCircuit size={12} className="text-[#f72585]" /> STORM_<span className="text-[#f72585]">ACTIVE</span>
            </span>
          </div>
          {isSearching && (
            <div className="flex items-center gap-2 text-[#b5179e] font-pixel text-xs animate-pulse tracking-widest uppercase">
              <Zap size={14} /> RUNNING
            </div>
          )}
        </header>

        {/* chat */}
        <div className="flex-1 overflow-y-auto px-4 py-8 md:px-12 space-y-8 z-10 pb-40">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto">
              <div className="pixel-panel p-10 w-full border-[#b5179e]">
                <h2 className="text-4xl font-pixel mb-6 flex justify-center items-center gap-4 neon-text-purple uppercase tracking-widest">
                  <Image src="/logo.png" alt="ixchio logo" width={48} height={48} className="pixelated" />
                  IXCHIO
                </h2>
                <div className="text-zinc-400 font-mono text-xs leading-relaxed mb-8 border-l-2 border-[#b5179e] pl-5 bg-black/40 py-3 pr-3">
                  <span className="text-[#f72585]">&gt; IXCHIO_DEEP_RESEARCH v3.0</span><br />
                  &gt; STORM perspectives + Reflection loops<br />
                  &gt; Adaptive search: Tavily + Jina<br />
                  <span className="animate-pulse">&gt; awaiting query_</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto space-y-10">
              <AnimatePresence>
                {messages.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {msg.role === "user" ? (
                      <div className="max-w-[85%] pixel-panel !border-[#4cc9f0] p-5 text-[#4cc9f0] font-mono text-sm relative">
                        <div className="absolute top-0 right-0 w-2 h-2 bg-[#4cc9f0]" />
                        <span className="font-pixel uppercase opacity-50 block mb-2 text-[10px] tracking-widest">&gt; QUERY</span>
                        {msg.content}
                      </div>
                    ) : msg.status !== "completed" ? (
                      <div className="font-mono text-xs text-[#f72585] bg-zinc-900/60 p-4 border-l-2 border-[#f72585] flex items-center justify-between w-full max-w-[95%]">
                        <span className="animate-pulse">{msg.content}</span>
                        <span className="font-pixel tracking-widest">{msg.progress}%</span>
                      </div>
                    ) : (
                      <div className="w-full max-w-[95%] pixel-panel p-8 prose prose-invert prose-p:text-zinc-300 prose-headings:font-pixel prose-headings:uppercase prose-headings:tracking-widest prose-a:text-[#4cc9f0] prose-strong:text-[#b5179e] max-w-none">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
              <div ref={endRef} className="h-4" />
            </div>
          )}
        </div>

        {/* input */}
        <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black via-black/95 to-transparent z-30">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={submit} className="relative flex items-center">
              <div className="absolute left-5 top-1/2 -translate-y-1/2 text-[#b5179e] font-pixel text-2xl animate-pulse">&gt;</div>
              <input
                type="text" value={query} onChange={e => setQuery(e.target.value)}
                placeholder="ENTER RESEARCH QUERY..." disabled={isSearching} spellCheck={false}
                className="w-full bg-[#050505] border-2 border-zinc-800 focus:border-[#b5179e] text-white font-mono text-sm pl-12 pr-16 py-5 outline-none transition-all disabled:opacity-50"
              />
              <button type="submit" disabled={!query.trim() || isSearching}
                className="absolute right-4 top-1/2 -translate-y-1/2 p-2 text-[#b5179e] disabled:opacity-50"
              >
                <Send size={20} />
              </button>
            </form>
            <div className="text-center mt-3 text-[9px] uppercase tracking-[0.3em] text-zinc-600 font-pixel flex items-center justify-center gap-4">
              <span>IXCHIO v3.0</span>
              <span className="w-1 h-1 bg-zinc-600 rounded-full" />
              <span>JWT SECURED</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
