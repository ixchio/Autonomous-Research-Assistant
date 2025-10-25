"""
Multi-Agent System for Research Assistant
File: main.py
All agents use free APIs with proper rate limiting (NO GEMINI)
"""

from dotenv import load_dotenv
load_dotenv()
import os
# i put here for reson to check
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from bs4 import BeautifulSoup
import tiktoken
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import uvicorn
import uuid
from enum import Enum
from collections import defaultdict
from time import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Rate Limiter ---

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            'groq': {'rpm': 30, 'rpd': 14400, 'tpm': 6000},
            'openrouter': {'rpm': 20, 'rpd': 200},  # Conservative for free tier
            'tavily': {'rpm': 5, 'rpd': 100}  # Very conservative
        }
    
    def can_request(self, service: str) -> (bool, int):
        now = time()
        minute_ago = now - 60
        day_ago = now - 86400
        
        # Clean old requests
        self.requests[service] = [t for t in self.requests[service] if t > day_ago]
        
        # Check limits
        limits = self.limits.get(service, {'rpm': 10, 'rpd': 1000})
        recent_minute = [t for t in self.requests[service] if t > minute_ago]
        recent_day = self.requests[service]
        
        rpm_limit = limits.get('rpm', float('inf'))
        rpd_limit = limits.get('rpd', float('inf'))
        
        if len(recent_minute) >= rpm_limit:
            wait_time = 60 - (now - recent_minute[0])
            return False, int(wait_time) + 1
            
        if len(recent_day) >= rpd_limit:
            wait_time = 86400 - (now - recent_day[0])
            return False, int(wait_time) + 1
        
        return True, 0
    
    async def wait_if_needed(self, service: str, max_wait: int = 120):
        """Wait if rate limit is hit"""
        total_waited = 0
        while True:
            can_go, wait_time = self.can_request(service)
            if can_go:
                self.requests[service].append(time())
                break
            
            if total_waited + wait_time > max_wait:
                raise HTTPException(status_code=429, detail=f"Rate limit exceeded for {service}. Try again later.")
            
            print(f"⏳ Rate limiting {service}, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            total_waited += wait_time

rate_limiter = RateLimiter()

# --- API Clients ---

class GroqClient:
    """Groq API client with rate limiting"""
    def __init__(self, rate_limiter):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.rate_limiter = rate_limiter
        self.model = "llama-3.1-8b-instant"
        
        if not self.api_key:
            print("❌ GROQ_API_KEY environment variable not set. GroqClient will fail.")
    
    async def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1500):
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
            
        await self.rate_limiter.wait_if_needed('groq')
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": min(max_tokens, 1500)
            }
            
            async with session.post(self.base_url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await resp.text()
                    raise Exception(f"Groq API error: {resp.status} - {error_text}")

class OpenRouterClient:
    """OpenRouter API client for free models"""
    def __init__(self, rate_limiter):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.rate_limiter = rate_limiter
        self.model = "meta-llama/llama-3.2-3b-instruct:free"
        
        if not self.api_key:
            print("❌ OPENROUTER_API_KEY environment variable not set. OpenRouterClient will fail.")
    
    async def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000):
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        await self.rate_limiter.wait_if_needed('openrouter')
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://research-assistant.app",
                "X-Title": "Autonomous Research Assistant"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            async with session.post(self.base_url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content']
                else:
                    error_text = await resp.text()
                    raise Exception(f"OpenRouter API error: {resp.status} - {error_text}")

class TavilyClient:
    """Tavily Search API client"""
    def __init__(self, rate_limiter):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https.api.tavily.com/search"
        self.rate_limiter = rate_limiter
        
        if not self.api_key:
            print("❌ TAVILY_API_KEY environment variable not set. TavilyClient will fail.")
    
    async def search(self, query: str, max_results: int = 5):
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")
            
        await self.rate_limiter.wait_if_needed('tavily')
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False
            }
            
            async with session.post(self.base_url, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    raise Exception(f"Tavily API error: {resp.status} - {error_text}")

# --- Vector DB ---

class VectorDB:
    """FAISS-based vector database for embeddings"""
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, runs locally
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadata = []
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the vector database"""
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(texts)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))
    
    def search(self, query: str, k: int = 5):
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            min(k, self.index.ntotal)
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'score': float(distances[0][i])
                })
        
        return results

# --- Agents ---

class ResearchPlannerAgent:
    """Breaks down research queries into subtasks using Groq"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.client = GroqClient(rate_limiter)
    
    async def create_plan(self, query: str, depth: str) -> Dict[str, Any]:
        """Create a structured research plan"""
        
        num_queries = {'quick': 3, 'medium': 5, 'deep': 7}.get(depth, 5)
        
        prompt = f"""You are a research planning expert. Break down this research query into a structured plan.

Research Query: {query}
Research Depth: {depth}

Create a detailed research plan with:
1. Main research objectives (2-3 objectives)
2. Key search queries to execute ({num_queries} specific queries)
3. Expected information types to gather
4. Validation criteria for quality

Format as JSON with keys: objectives, search_queries, info_types, validation_criteria
Be specific and actionable. Keep it concise."""

        try:
            response = await self.client.chat([
                {"role": "system", "content": "You are a research planning expert. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ], temperature=0.3, max_tokens=1000)
            
            # Parse JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            plan = json.loads(response.strip())
            
            if 'search_queries' in plan:
                plan['search_queries'] = plan['search_queries'][:num_queries]
            
            return plan
            
        except Exception as e:
            print(f"❌ Planning error: {str(e)}, using fallback plan")
            return {
                "objectives": [query],
                "search_queries": [query][:num_queries],
                "info_types": ["facts", "statistics", "expert opinions"],
                "validation_criteria": ["accuracy", "relevance", "recency"]
            }

class WebSearchAgent:
    """Executes web searches using Tavily"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.tavily = TavilyClient(rate_limiter)
    
    async def search(self, research_plan: Dict, max_sources: int) -> List[Dict]:
        """Execute searches based on research plan"""
        search_queries = research_plan.get('search_queries', [])
        all_results = []
        
        max_api_calls = min(5, max_sources // 3)
        queries_to_execute = search_queries[:max_api_calls]
        
        print(f"🔍 Executing {len(queries_to_execute)} search queries...")
        
        for idx, query in enumerate(queries_to_execute, 1):
            try:
                print(f"  [{idx}/{len(queries_to_execute)}] Searching: {query[:50]}...")
                results = await self.tavily.search(query, max_results=3)
                
                for result in results.get('results', []):
                    all_results.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'content': result.get('content', ''),
                        'score': result.get('score', 0),
                        'query': query,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Search error for query '{query[:30]}...': {str(e)}")
                continue
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:max_sources]

class DataExtractionAgent:
    """Extracts relevant facts and information from search results"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.client = GroqClient(rate_limiter)
    
    async def extract(self, search_results: List[Dict], research_plan: Dict) -> List[Dict]:
        """Extract structured information from search results"""
        extracted_data = []
        
        objectives = research_plan.get('objectives', [])
        objectives_text = "\n".join(f"- {obj}" for obj in objectives)
        
        batch_size = 3
        max_batches = 5
        batches_processed = 0
        
        for i in range(0, len(search_results), batch_size):
            if batches_processed >= max_batches:
                break
                
            batch = search_results[i:i + batch_size]
            
            content_parts = []
            for idx, result in enumerate(batch):
                content_parts.append(
                    f"Source {idx + 1}: {result['title']}\n"
                    f"Content: {result['content'][:400]}...\n"
                )
            
            prompt = f"""Extract key facts and insights from these sources.

Research Objectives:
{objectives_text}

Sources:
{chr(10).join(content_parts)}

For each relevant fact, provide:
1. The fact/insight (concise)
2. Source reference (Source 1, 2, or 3)
3. Relevance score (1-10)

Format as JSON array: [{{"fact": "...", "source_idx": 1, "relevance": 8}}, ...]
Maximum 10 facts total."""

            try:
                response = await self.client.chat([
                    {"role": "system", "content": "You are a precise data extraction expert. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ], temperature=0.2, max_tokens=1200)
                
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                
                facts = json.loads(response.strip())
                
                for fact in facts:
                    if isinstance(fact, dict):
                        source_idx_val = fact.get('source_idx', 1)
                        try:
                            source_idx = int(source_idx_val) - 1
                        except ValueError:
                            source_idx = 0
                            
                        if 0 <= source_idx < len(batch):
                            fact['url'] = batch[source_idx]['url']
                            fact['title'] = batch[source_idx]['title']
                        extracted_data.append(fact)
                
                batches_processed += 1
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Extraction error: {str(e)}")
                for result in batch:
                    extracted_data.append({
                        'fact': result['content'][:200],
                        'url': result['url'],
                        'title': result['title'],
                        'relevance': 5
                    })
                batches_processed += 1
        
        return extracted_data

class SynthesizerAgent:
    """Combines information from multiple sources using OpenRouter"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.client = OpenRouterClient(rate_limiter)
        self.vector_db = VectorDB()
    
    async def synthesize(self, extracted_data: List[Dict], research_plan: Dict, query: str) -> Dict:
        """Synthesize information into coherent insights"""
        
        texts = [fact['fact'] for fact in extracted_data if 'fact' in fact]
        metadata = [{'url': fact.get('url', ''), 'title': fact.get('title', '')} 
                   for fact in extracted_data if 'fact' in fact]
        
        if texts:
            self.vector_db.add_documents(texts, metadata)
        
        relevant_facts = self.vector_db.search(query, k=min(15, len(texts)))
        
        facts_text = "\n".join([
            f"- {fact['text']}"
            for fact in relevant_facts[:12]
        ])
        
        prompt = f"""Synthesize these research findings into coherent insights.

Research Query: {query}

Key Facts:
{facts_text}

Create a concise synthesis that:
1. Identifies main themes (2-3 themes)
2. Highlights key insights (3-5 insights)
3. Notes any important patterns or contradictions

Keep it clear and focused. Maximum 300 words."""

        try:
            synthesis = await self.client.chat([
                {"role": "system", "content": "You are a research synthesis expert."},
                {"role": "user", "content": prompt}
            ], temperature=0.5, max_tokens=1500)
            
            return {
                'synthesis': synthesis,
                'fact_count': len(relevant_facts),
                'sources_used': len(set(f['metadata'].get('url', '') for f in relevant_facts))
            }
        except Exception as e:
            print(f"❌ Synthesis error: {str(e)}")
            return {
                'synthesis': f"Research findings on '{query}':\n\n" + "\n".join([f"• {f['text']}" for f in relevant_facts[:10]]),
                'fact_count': len(relevant_facts),
                'sources_used': len(set(f['metadata'].get('url', '') for f in relevant_facts))
            }

class ReportWriterAgent:
    """Generates formatted research reports using Groq"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.client = GroqClient(rate_limiter)
    
    async def write_report(
        self, 
        synthesized_content: Dict, 
        research_plan: Dict,
        sources: List[Dict],
        include_citations: bool
    ) -> str:
        """Generate final research report"""
        
        synthesis_text = synthesized_content.get('synthesis', '')
        objectives = research_plan.get('objectives', [])
        
        prompt = f"""Write a comprehensive research report.

Research Objectives:
{chr(10).join(f'{i+1}. {obj}' for i, obj in enumerate(objectives[:3]))}

Synthesized Findings:
{synthesis_text[:1500]}

Write a well-structured report with:
# Executive Summary
# Key Findings  
# Analysis
# Conclusions

Use clear headings and professional tone. Maximum 500 words."""

        try:
            report = await self.client.chat([
                {"role": "system", "content": "You are an expert research report writer."},
                {"role": "user", "content": prompt}
            ], temperature=0.6, max_tokens=1500)
            
            if include_citations and sources:
                citations = "\n\n## Sources\n\n"
                unique_sources = {s['url']: s for s in sources}.values()
                for i, source in enumerate(list(unique_sources)[:15], 1):
                    citations += f"{i}. [{source['title']}]({source['url']})\n"
                
                report += citations
            
            return report
            
        except Exception as e:
            print(f"❌ Report writing error: {str(e)}")
            return f"# Research Report\n\n{synthesis_text}\n\n## Note\nFull report generation encountered an error."

class QualityCheckerAgent:
    """Validates report quality - Simple rule-based to save API calls"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
    
    async def validate(self, report: str, sources: List[Dict], extracted_data: List[Dict]) -> Dict:
        """Validate research report quality"""
        
        word_count = len(report.split())
        has_structure = any(marker in report for marker in ['Summary', 'Findings', 'Executive', 'Analysis'])
        has_content = word_count > 200
        source_count = len(sources)
        
        is_valid = has_content and has_structure and source_count > 0
        
        if is_valid:
            score = min(10, 5 + (word_count // 100) + (source_count // 2))
        else:
            score = 3
        
        return {
            'is_valid': is_valid,
            'score': score,
            'issues': [] if is_valid else ['Insufficient content or structure'],
            'reason': 'Quality validation passed' if is_valid else 'Quality checks failed',
            'metrics': {
                'word_count': word_count,
                'source_count': source_count,
                'has_structure': has_structure
            }
        }

# --- API Data Models ---

class ResearchStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    SEARCHING = "searching"
    EXTRACTING = "extracting"
    SYNTHESIZING = "synthesizing"
    WRITING = "writing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=500)
    depth: str = Field(default="medium", pattern="^(quick|medium|deep)$")
    max_sources: int = Field(default=10, ge=5, le=30)
    include_citations: bool = True

class ResearchResponse(BaseModel):
    task_id: str
    status: ResearchStatus
    message: str

class ResearchResult(BaseModel):
    task_id: str
    query: str
    status: ResearchStatus
    progress: int
    current_step: str
    report: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


# --- Global Task Storage ---
research_tasks: Dict[str, ResearchResult] = {}


# --- FastAPI App Definition ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("🚀 Autonomous Research Assistant API started")
    print("🤖 Multi-agent system initialized (NO GEMINI)")
    print("💰 Free tier resources configured")
    print("📊 Rate limits: Groq=30RPM, OpenRouter=20RPM, Tavily=5RPM")
    
    yield  # Application runs here
    
    # Code to run on shutdown
    print("👋 Shutting down gracefully")

# THIS IS THE LINE THAT WAS MISSING 
app = FastAPI(
    title="Autonomous Research Assistant API",
    description="AI-powered research assistant with multi-agent architecture (No Gemini)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "name": "Autonomous Research Assistant API",
        "version": "2.0.0",
        "status": "operational",
        "models": {
            "planner": "Groq (Llama 3.1)",
            "synthesizer": "OpenRouter (Llama 3.2)",
            "writer": "Groq (Llama 3.1)",
            "quality_checker": "Rule-based"
        },
        "endpoints": {
            "research": "/api/v1/research",
            "status": "/api/v1/research/{task_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_tasks": len([t for t in research_tasks.values() if t.status not in [ResearchStatus.COMPLETED, ResearchStatus.FAILED]]),
        "total_tasks": len(research_tasks)
    }

@app.post("/api/v1/research", response_model=ResearchResponse)
async def create_research_task(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
):
    """Create a new research task"""
    task_id = str(uuid.uuid4())
    
    research_tasks[task_id] = ResearchResult(
        task_id=task_id,
        query=request.query,
        status=ResearchStatus.PENDING,
        progress=0,
        current_step="Initializing research task",
        created_at=datetime.utcnow().isoformat(),
        metadata={
            "depth": request.depth,
            "max_sources": request.max_sources,
            "include_citations": request.include_citations
        }
    )
    
    background_tasks.add_task(
        execute_research_pipeline,
        task_id,
        request
    )
    
    return ResearchResponse(
        task_id=task_id,
        status=ResearchStatus.PENDING,
        message="Research task created successfully"
    )

@app.get("/api/v1/research/{task_id}", response_model=ResearchResult)
async def get_research_status(task_id: str):
    """Get the status and results of a research task"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Research task not found")
    
    return research_tasks[task_id]

@app.get("/api/v1/research", response_model=List[ResearchResult])
async def list_research_tasks(
    status: Optional[ResearchStatus] = None,
    limit: int = 20
):
    """List research tasks with optional filtering"""
    tasks = list(research_tasks.values())
    
    if status:
        tasks = [t for t in tasks if t.status == status]
    
    tasks.sort(key=lambda x: x.created_at, reverse=True)
    
    return tasks[:limit]

@app.delete("/api/v1/research/{task_id}")
async def delete_research_task(task_id: str):
    """Delete a research task"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Research task not found")
    
    del research_tasks[task_id]
    return {"message": "Research task deleted successfully"}

# --- Research Pipeline Orchestrator ---

async def execute_research_pipeline(task_id: str, request: ResearchRequest):
    """Main research pipeline orchestrator"""
    task = research_tasks[task_id]
    
    try:
        # Initialize agents
        planner = ResearchPlannerAgent(rate_limiter)
        searcher = WebSearchAgent(rate_limiter)
        extractor = DataExtractionAgent(rate_limiter)
        synthesizer = SynthesizerAgent(rate_limiter)
        writer = ReportWriterAgent(rate_limiter)
        checker = QualityCheckerAgent(rate_limiter)
        
        # Step 1: Planning
        task.status = ResearchStatus.PLANNING
        task.current_step = "Creating research plan"
        task.progress = 10
        print(f"📋 [{task_id[:8]}] Planning research...")
        
        research_plan = await planner.create_plan(request.query, request.depth)
        task.metadata['plan'] = research_plan
        
        # Step 2: Web Search
        task.status = ResearchStatus.SEARCHING
        task.current_step = "Searching for information"
        task.progress = 25
        print(f"🔍 [{task_id[:8]}] Searching web...")
        
        search_results = await searcher.search(research_plan, request.max_sources)
        task.sources = search_results
        task.progress = 40
        
        if not search_results:
            raise Exception("No search results found")
        
        # Step 3: Data Extraction
        task.status = ResearchStatus.EXTRACTING
        task.current_step = "Extracting relevant information"
        task.progress = 50
        print(f"📊 [{task_id[:8]}] Extracting data...")
        
        extracted_data = await extractor.extract(search_results, research_plan)
        task.metadata['extracted_facts'] = len(extracted_data)
        task.progress = 60
        
        if not extracted_data:
            raise Exception("No data could be extracted from sources")
        
        # Step 4: Synthesis
        task.status = ResearchStatus.SYNTHESIZING
        task.current_step = "Synthesizing information"
        task.progress = 70
        print(f"🧠 [{task_id[:8]}] Synthesizing findings...")
        
        synthesized_content = await synthesizer.synthesize(
            extracted_data,
            research_plan,
            request.query
        )
        task.progress = 80
        
        # Step 5: Report Writing
        task.status = ResearchStatus.WRITING
        task.current_step = "Writing research report"
        task.progress = 85
        print(f"✍️ [{task_id[:8]}] Writing report...")
        
        report = await writer.write_report(
            synthesized_content,
            research_plan,
            task.sources,
            request.include_citations
        )
        task.report = report
        task.progress = 95
        
        # Step 6: Quality Check
        task.status = ResearchStatus.VALIDATING
        task.current_step = "Validating report quality"
        print(f"✅ [{task_id[:8]}] Validating quality...")
        
        validation_result = await checker.validate(
            report,
            task.sources,
            extracted_data
        )
        
        if validation_result['is_valid']:
            task.status = ResearchStatus.COMPLETED
            task.current_step = "Research completed successfully"
            task.progress = 100
            task.completed_at = datetime.utcnow().isoformat()
            task.metadata['quality_score'] = validation_result.get('score', 'N/A')
            print(f"🎉 [{task_id[:8]}] Research completed! Score: {validation_result.get('score')}")
        else:
            task.status = ResearchStatus.FAILED
            task.error = f"Quality validation failed: {validation_result.get('reason', 'Unknown')}"
            task.progress = 95
            print(f"⚠️ [{task_id[:8]}] Quality check failed")
        
    except Exception as e:
        print(f"❌ [{task_id[:8]}] Error in research pipeline: {str(e)}")
        task.status = ResearchStatus.FAILED
        task.error = str(e)
        task.current_step = f"Research failed: {str(e)}"

# --- Local Execution ---

if __name__ == "__main__":
    # This block is for local development.
    # Render will use the 'app' object directly with uvicorn.
    print("Starting server for local development...")
    uvicorn.run(
        "main:app",  # Refers to this file (main.py) and the 'app' object
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enables auto-reload on code changes
        log_level="info"
    )
