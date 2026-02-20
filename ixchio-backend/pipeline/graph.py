"""
Deep Research Graph — the brain of ixchio.

Pipeline flow:
  plan (Cerebras) → STORM perspectives → adaptive search (Tavily/Jina) →
  deep extract (Jina Reader) → synthesize (BM25 + Vector) → write report →
  reflect (self-critique) → [follow-up search if gaps] → validate (3-model consensus)

Each node is a pure async function that takes ResearchState and returns ResearchState.
LangGraph handles the wiring and conditional routing.
"""

import asyncio
import uuid
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi

from langgraph.graph import StateGraph, END

from pipeline.state import ResearchState
from core.helpers import extract_json
from core.cache import SemanticCache
from core.circuit_breaker import CircuitBreaker
from core.rate_limiter import RateLimiter
from core.vector_db import PersistentVectorDB

from clients import GroqClient, OpenRouterClient, TavilyClient, CerebrasClient, JinaClient


class DeepResearchGraph:
    def __init__(self, vector_db=None):
        rl = RateLimiter()

        self.groq = GroqClient(rl, CircuitBreaker())
        self.openrouter = OpenRouterClient(rl, CircuitBreaker())
        self.tavily = TavilyClient(rl, CircuitBreaker())
        self.cerebras = CerebrasClient(rl, CircuitBreaker())
        self.jina = JinaClient(rl, CircuitBreaker())

        self.cache = SemanticCache()
        self.vector_db = vector_db or PersistentVectorDB()

        self.graph = self._wire_graph()

    def _wire_graph(self) -> StateGraph:
        g = StateGraph(ResearchState)

        g.add_node("plan", self._plan)
        g.add_node("storm", self._storm_perspectives)
        g.add_node("search", self._adaptive_search)
        g.add_node("extract", self._deep_extract)
        g.add_node("synthesize", self._synthesize)
        g.add_node("write", self._write_report)
        g.add_node("reflect", self._reflect)
        g.add_node("followup", self._followup_search)
        g.add_node("validate", self._validate)

        g.set_entry_point("plan")

        g.add_edge("plan", "storm")
        g.add_edge("storm", "search")
        g.add_conditional_edges("search", self._retry_or_proceed, {
            "retry": "search", "proceed": "extract"
        })
        g.add_edge("extract", "synthesize")
        g.add_edge("synthesize", "write")
        g.add_edge("write", "reflect")
        g.add_conditional_edges("reflect", self._needs_followup, {
            "followup": "followup", "done": "validate"
        })
        g.add_edge("followup", "write")
        g.add_conditional_edges("validate", self._accept_or_redo, {
            "accept": END, "redo": "write", "fail": END
        })

        return g.compile()

    # ------------------------------------------------------------------
    # helpers for routing
    # ------------------------------------------------------------------
    def _retry_or_proceed(self, state: ResearchState) -> str:
        if not state["search_results"] and state["retry_count"] <= 2:
            return "retry"
        return "proceed"

    def _needs_followup(self, state: ResearchState) -> str:
        gaps = state.get("reflection_gaps") or []
        if gaps and state["reflection_count"] <= 2:
            return "followup"
        return "done"

    def _accept_or_redo(self, state: ResearchState) -> str:
        word_count = len((state.get("report") or "").split())
        if word_count < 100 and state["retry_count"] <= 2:
            return "redo"
        if word_count < 50:
            return "fail"
        return "accept"

    # ------------------------------------------------------------------
    # NODE: Plan — Cerebras speed brain decomposes the query
    # ------------------------------------------------------------------
    async def _plan(self, state: ResearchState) -> ResearchState:
        print(f"⚡ [plan] breaking down: {state['query'][:60]}...")

        async def _do_plan():
            prompt = (
                f"Break this research query into 5 diverse sub-queries. "
                f"Classify each as 'news', 'technical', or 'general'.\n"
                f"Query: {state['query']}\n"
                f'Return JSON: {{"search_queries": [{{"query": "...", "type": "news|technical|general"}}]}}'
            )
            msgs = [
                {"role": "system", "content": "Research planner. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ]
            try:
                raw = await self.cerebras.chat(msgs, temperature=0.3)
            except Exception:
                raw = await self.groq.chat(msgs, temperature=0.3)
            return extract_json(raw)

        plan, cache_status = await self.cache.get_or_compute(
            f"plan:{state['query']}", _do_plan
        )
        state["research_plan"] = plan
        state["progress"] = 10
        if cache_status == "cache_hit":
            state["cache_hits"] += 1
        else:
            state["total_api_calls"] += 1
        return state

    # ------------------------------------------------------------------
    # NODE: STORM — generate expert personas who question the topic
    # ------------------------------------------------------------------
    async def _storm_perspectives(self, state: ResearchState) -> ResearchState:
        print("🧑‍🔬 [storm] generating expert panel...")

        prompt = (
            f"Topic: '{state['query']}'\n"
            f"Create 3 expert personas with *different* angles on this topic.\n"
            f"Each has: name, expertise, and 2 sharp questions they'd ask.\n"
            f'Return JSON: {{"experts": [{{"name": "...", "expertise": "...", "questions": ["...", "..."]}}]}}'
        )
        msgs = [
            {"role": "system", "content": "Academic panel organizer. JSON only."},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await self.cerebras.chat(msgs, temperature=0.7)
        except Exception:
            raw = await self.groq.chat(msgs, temperature=0.7)

        parsed = extract_json(raw)
        experts = parsed.get("experts", [])
        state["expert_perspectives"] = experts

        # fold expert questions into the search plan
        extra_qs = []
        for expert in experts:
            for q in expert.get("questions", []):
                extra_qs.append({"query": q, "type": "general"})

        existing = state["research_plan"].get("search_queries", [])
        # handle case where planner returned plain strings instead of dicts
        if existing and isinstance(existing[0], str):
            existing = [{"query": q, "type": "general"} for q in existing]

        state["research_plan"]["search_queries"] = existing + extra_qs[:4]
        state["progress"] = 15
        state["total_api_calls"] += 1
        return state

    # ------------------------------------------------------------------
    # NODE: Adaptive Search — route each query to the right engine
    # ------------------------------------------------------------------
    async def _adaptive_search(self, state: ResearchState) -> ResearchState:
        print(f"🔍 [search] round {state['search_round']}...")

        queries = state["research_plan"].get(
            "search_queries", [{"query": state["query"], "type": "general"}]
        )
        if queries and isinstance(queries[0], str):
            queries = [{"query": q, "type": "general"} for q in queries]

        results = state.get("search_results") or []

        for item in queries[:8]:
            q = item["query"] if isinstance(item, dict) else str(item)
            q_type = item.get("type", "general") if isinstance(item, dict) else "general"

            try:
                if q_type == "technical":
                    # jina for docs / papers
                    hits = await self.jina.search(q)
                    for h in hits:
                        results.append({**h, "source_engine": "jina"})
                else:
                    # tavily for news / general
                    resp = await self.tavily.search(q, max_results=3)
                    for r in resp.get("results", []):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "content": r.get("content", ""),
                            "score": r.get("score", 0),
                            "source_engine": "tavily",
                        })
                state["total_api_calls"] += 1
            except Exception as e:
                state["errors"].append(f"search/{q_type}: {e}")

        state["search_results"] = results
        if not results:
            state["retry_count"] += 1
        state["progress"] = 30
        return state

    # ------------------------------------------------------------------
    # NODE: Deep Extract — Jina Reader pulls full page content
    # ------------------------------------------------------------------
    async def _deep_extract(self, state: ResearchState) -> ResearchState:
        print(f"📖 [extract] pulling top sources via Jina Reader...")

        # pick the top-scoring unique URLs
        seen, top_urls = set(), []
        for r in sorted(state["search_results"], key=lambda x: x.get("score", 0), reverse=True):
            url = r.get("url", "")
            if url and url not in seen and len(top_urls) < 5:
                top_urls.append(url)
                seen.add(url)

        # basic extraction from search snippets
        extracted = []
        for r in state["search_results"][:10]:
            extracted.append({
                "fact": r.get("content", "")[:300],
                "url": r.get("url", ""),
                "title": r.get("title", ""),
            })

        # deep pull via Jina Reader (top 3)
        deep = []
        for url in top_urls[:3]:
            try:
                content = await self.jina.read_url(url)
                deep.append({"url": url, "full_content": content[:3000]})
                state["total_api_calls"] += 1
            except Exception as e:
                state["errors"].append(f"jina_reader: {e}")

        state["extracted_data"] = extracted
        state["deep_extractions"] = deep
        state["progress"] = 50
        return state

    # ------------------------------------------------------------------
    # NODE: Synthesize — BM25 + Vector hybrid fusion
    # ------------------------------------------------------------------
    async def _synthesize(self, state: ResearchState) -> ResearchState:
        texts = [f["fact"] for f in state["extracted_data"]]

        # add deep extraction content as bonus evidence
        for de in (state.get("deep_extractions") or []):
            texts.append(de["full_content"][:500])

        metadata = [
            {"url": f.get("url", ""), "title": f.get("title", "")}
            for f in state["extracted_data"]
        ]

        if not texts:
            state["synthesized_content"] = {"key_facts": [], "fact_count": 0}
            state["progress"] = 65
            return state

        # store in vector DB for future retrieval
        self.vector_db.add_documents(texts[: len(metadata)], metadata)

        # vector search
        vec_hits = self.vector_db.search(state["query"], k=10)
        vec_texts = [h["text"] for h in vec_hits]

        # BM25
        tokenized = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(state["query"].split())
        top_idx = np.argsort(scores)[-10:][::-1]
        bm25_texts = [texts[i] for i in top_idx if scores[i] > 0]

        # fuse + dedup
        combined, seen = [], set()
        for t in vec_texts + bm25_texts:
            if t not in seen:
                seen.add(t)
                combined.append(t)

        state["synthesized_content"] = {
            "key_facts": combined[:12],
            "fact_count": len(combined[:12]),
        }
        state["progress"] = 65
        return state

    # ------------------------------------------------------------------
    # NODE: Write Report — weaves in expert perspectives + gap fixes
    # ------------------------------------------------------------------
    async def _write_report(self, state: ResearchState) -> ResearchState:
        print("✍️ [write] drafting report...")

        facts = "\n".join(state["synthesized_content"]["key_facts"][:10])

        expert_block = ""
        for ex in (state.get("expert_perspectives") or []):
            qs = ", ".join(ex.get("questions", []))
            expert_block += f"\n- {ex.get('name', 'Expert')} ({ex.get('expertise', '')}): {qs}"

        gap_block = ""
        if state.get("reflection_gaps"):
            gap_block = "\n\nPrevious draft had gaps — address these:\n"
            gap_block += "\n".join(f"- {g}" for g in state["reflection_gaps"])

        prompt = (
            f"Write a comprehensive research report on: {state['query']}\n\n"
            f"Key findings:\n{facts}\n\n"
            f"Expert perspectives to consider:{expert_block}\n"
            f"{gap_block}\n\n"
            f"Structure: Executive Summary, Key Findings (cite sources), "
            f"Perspectives Analysis, Conclusions. Use markdown."
        )

        report = await self.groq.chat(
            [
                {"role": "system", "content": "Senior research analyst. Thorough, well-cited."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
        )

        state["report"] = report
        state["progress"] = 80
        state["total_api_calls"] += 1
        return state

    # ------------------------------------------------------------------
    # NODE: Reflect — self-critique the draft
    # ------------------------------------------------------------------
    async def _reflect(self, state: ResearchState) -> ResearchState:
        round_num = state["reflection_count"] + 1
        print(f"🤔 [reflect] round {round_num}...")

        prompt = (
            f"You're a tough reviewer. Read this report on '{state['query']}' "
            f"and find 2-3 factual gaps or weak claims.\n\n"
            f"Report:\n{state['report'][:2500]}\n\n"
            f'Return JSON: {{"gaps": ["...", "..."]}}\n'
            f'If it\'s solid, return {{"gaps": []}}'
        )

        try:
            raw = await self.cerebras.chat([{"role": "user", "content": prompt}], temperature=0.2)
        except Exception:
            raw = await self.groq.chat([{"role": "user", "content": prompt}], temperature=0.2)

        parsed = extract_json(raw)
        state["reflection_gaps"] = parsed.get("gaps", [])
        state["reflection_count"] = round_num
        state["total_api_calls"] += 1
        state["progress"] = 85
        return state

    # ------------------------------------------------------------------
    # NODE: Follow-up Search — targeted gap-filling
    # ------------------------------------------------------------------
    async def _followup_search(self, state: ResearchState) -> ResearchState:
        gaps = state.get("reflection_gaps", [])
        print(f"🔄 [followup] filling {len(gaps)} gaps...")

        new_facts = []
        for gap in gaps[:3]:
            try:
                resp = await self.tavily.search(f"{state['query']} {gap}", max_results=2)
                for r in resp.get("results", []):
                    new_facts.append(r.get("content", "")[:300])
                state["total_api_calls"] += 1
            except Exception as e:
                state["errors"].append(f"followup: {e}")

        existing = state["synthesized_content"].get("key_facts", [])
        state["synthesized_content"]["key_facts"] = existing + new_facts
        state["synthesized_content"]["fact_count"] = len(state["synthesized_content"]["key_facts"])
        state["search_round"] += 1
        state["progress"] = 70
        return state

    # ------------------------------------------------------------------
    # NODE: Validate — 3-model consensus vote
    # ------------------------------------------------------------------
    async def _validate(self, state: ResearchState) -> ResearchState:
        report = state.get("report", "")
        if len(report.split()) < 100:
            state["retry_count"] += 1
            state["progress"] = 100
            return state

        prompt = (
            f"Rate this report for '{state['query']}' on relevancy and consistency (0-10).\n\n"
            f"Report:\n{report[:2000]}\n\n"
            f'Return JSON: {{"relevancy": N, "consistency": N}}'
        )

        evals = [
            self.groq.chat([{"role": "user", "content": prompt}], temperature=0.1),
            self.openrouter.chat([{"role": "user", "content": prompt}], temperature=0.1),
        ]
        try:
            evals.append(self.cerebras.chat([{"role": "user", "content": prompt}], temperature=0.1))
        except Exception:
            evals.append(self.groq.chat([{"role": "user", "content": prompt}], temperature=0.5))

        results = await asyncio.gather(*evals, return_exceptions=True)

        scores = []
        for res in results:
            if not isinstance(res, Exception):
                parsed = extract_json(res)
                if "relevancy" in parsed:
                    scores.append(parsed)

        if scores:
            avg = sum(s.get("relevancy", 5) for s in scores) / len(scores)
            print(f"🔬 [validate] consensus: {avg:.1f}/10")
            if avg < 6.0:
                state["retry_count"] += 1

        state["progress"] = 100
        return state

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    async def run(self, query: str, depth: str = "medium", max_sources: int = 10, task_id: str = None) -> ResearchState:
        initial = ResearchState(
            query=query,
            depth=depth,
            max_sources=max_sources,
            task_id=task_id or str(uuid.uuid4()),
            research_plan=None,
            expert_perspectives=None,
            search_results=None,
            extracted_data=None,
            deep_extractions=None,
            synthesized_content=None,
            report=None,
            reflection_gaps=None,
            current_step="Initializing",
            progress=0,
            retry_count=0,
            search_round=1,
            reflection_count=0,
            errors=[],
            cache_hits=0,
            total_api_calls=0,
        )
        return await self.graph.ainvoke(initial)
