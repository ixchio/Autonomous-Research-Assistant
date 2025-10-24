"""
Multi-Agent System for Research Assistant
All agents use free APIs with proper rate limiting (Gemini removed)
"""

import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
from bs4 import BeautifulSoup
import tiktoken

#rate limiter here...brokee

class RateLimiter:
    """Manages rate limits for different API services"""
    def __init__(self):
        self.limits = {
            'groq': {'rpm': 30, 'rpd': 14400, 'tpm': 6000},  # Per minute limits
            'openrouter': {'rpm': 20, 'rpd': 200},  # Conservative free tier
            'tavily': {'rpm': 5, 'rpd': 100}  # Conservative for free tier
        }
        self.calls = {service: [] for service in self.limits.keys()}
        self.tokens = {'groq': []}
    
    async def wait_if_needed(self, service: str):
        """Wait if rate limit would be exceeded"""
        now = datetime.utcnow().timestamp()
        
        # Clean old calls (older than 1 minute)
        self.calls[service] = [t for t in self.calls[service] if now - t < 60]
        
        # Check RPM limit
        rpm_limit = self.limits[service]['rpm']
        if len(self.calls[service]) >= rpm_limit:
            oldest_call = self.calls[service][0]
            wait_time = 60 - (now - oldest_call)
            if wait_time > 0:
                print(f"⏳ Rate limit reached for {service}, waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time + 0.5)
        
        # Record this call
        self.calls[service].append(now)

class GroqClient:
    """Groq API client with rate limiting"""
    def __init__(self, rate_limiter):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.rate_limiter = rate_limiter
        self.model = "llama-3.1-8b-instant"  # 30 RPM, 14.4K RPD, 6K TPM
    
    async def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 1500):
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
                "max_tokens": min(max_tokens, 1500)  # Conservative for TPM limits
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
        # Using a reliable free model with good limits
        self.model = "meta-llama/llama-3.2-3b-instruct:free"  # 20 RPM free tier
    
    async def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000):
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
        self.base_url = "https://api.tavily.com/search"
        self.rate_limiter = rate_limiter
    
    async def search(self, query: str, max_results: int = 5):
        await self.rate_limiter.wait_if_needed('tavily')
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",  # Use basic to save on API calls
                "include_answer": True,
                "include_raw_content": False
            }
            
            async with session.post(self.base_url, json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    error_text = await resp.text()
                    raise Exception(f"Tavily API error: {resp.status} - {error_text}")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

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

class ResearchPlannerAgent:
    """Breaks down research queries into subtasks"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.client = GroqClient(rate_limiter)
    
    async def create_plan(self, query: str, depth: str) -> Dict[str, Any]:
        """Create a structured research plan"""
        
        # Adjust search queries based on depth
        num_queries = {'quick': 3, 'standard': 5, 'deep': 8}.get(depth, 5)
        
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
            
            # Ensure we don't exceed query limits
            if 'search_queries' in plan:
                plan['search_queries'] = plan['search_queries'][:num_queries]
            
            return plan
            
        except Exception as e:
            print(f"Planning error: {str(e)}, using fallback plan")
            # Fallback plan
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
        
        # Strict limit to avoid exceeding free tier
        max_api_calls = min(5, max_sources // 3)  # Max 5 API calls
        queries_to_execute = search_queries[:max_api_calls]
        
        print(f"🔍 Executing {len(queries_to_execute)} search queries...")
        
        for idx, query in enumerate(queries_to_execute, 1):
            try:
                print(f"  [{idx}/{len(queries_to_execute)}] Searching: {query[:50]}...")
                results = await self.tavily.search(query, max_results=3)  # Only 3 results per query
                
                for result in results.get('results', []):
                    all_results.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'content': result.get('content', ''),
                        'score': result.get('score', 0),
                        'query': query,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                
                # Delay between searches to respect rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Search error for query '{query[:30]}...': {str(e)}")
                continue
        
        # Sort by relevance score and limit
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
        
        # Process in batches to manage tokens and API calls
        batch_size = 3
        max_batches = 5  # Limit total API calls
        
        batches_processed = 0
        
        for i in range(0, len(search_results), batch_size):
            if batches_processed >= max_batches:
                break
                
            batch = search_results[i:i + batch_size]
            
            # Prepare content for extraction
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
                
                # Parse extracted data
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0]
                
                facts = json.loads(response.strip())
                
                # Add source URLs
                for fact in facts:
                    if isinstance(fact, dict):
                        source_idx = fact.get('source_idx', 1) - 1
                        if 0 <= source_idx < len(batch):
                            fact['url'] = batch[source_idx]['url']
                            fact['title'] = batch[source_idx]['title']
                        extracted_data.append(fact)
                
                batches_processed += 1
                await asyncio.sleep(0.5)  # Rate limit protection
                
            except Exception as e:
                print(f"Extraction error: {str(e)}")
                # Fallback: basic extraction
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
    """Combines information from multiple sources into coherent insights"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.client = OpenRouterClient(rate_limiter)
        self.vector_db = VectorDB()
    
    async def synthesize(self, extracted_data: List[Dict], research_plan: Dict, query: str) -> Dict:
        """Synthesize information into coherent insights"""
        
        # Add facts to vector database
        texts = [fact['fact'] for fact in extracted_data if 'fact' in fact]
        metadata = [{'url': fact.get('url', ''), 'title': fact.get('title', '')} 
                   for fact in extracted_data if 'fact' in fact]
        
        if texts:
            self.vector_db.add_documents(texts, metadata)
        
        # Retrieve most relevant facts
        relevant_facts = self.vector_db.search(query, k=min(15, len(texts)))
        
        # Prepare synthesis
        facts_text = "\n".join([
            f"- {fact['text']}"
            for fact in relevant_facts[:12]  # Limit for token efficiency
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
            print(f"Synthesis error: {str(e)}")
            # Simple fallback synthesis
            return {
                'synthesis': f"Research findings on '{query}':\n\n" + "\n".join([f"• {f['text']}" for f in relevant_facts[:10]]),
                'fact_count': len(relevant_facts),
                'sources_used': len(set(f['metadata'].get('url', '') for f in relevant_facts))
            }

class ReportWriterAgent:
    """Generates formatted research reports"""
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
            
            # Add citations if requested
            if include_citations and sources:
                citations = "\n\n## Sources\n\n"
                unique_sources = {s['url']: s for s in sources}.values()
                for i, source in enumerate(list(unique_sources)[:15], 1):
                    citations += f"{i}. [{source['title']}]({source['url']})\n"
                
                report += citations
            
            return report
            
        except Exception as e:
            print(f"Report writing error: {str(e)}")
            return f"# Research Report\n\n{synthesis_text}\n\n## Note\nFull report generation encountered an error."

class QualityCheckerAgent:
    """Validates report quality and accuracy"""
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.client = GroqClient(rate_limiter)
    
    async def validate(self, report: str, sources: List[Dict], extracted_data: List[Dict]) -> Dict:
        """Validate research report quality"""
        
        # Basic checks
        word_count = len(report.split())
        has_structure = any(marker in report for marker in ['#', 'Summary', 'Findings', 'Executive'])
        has_content = word_count > 200
        source_count = len(sources)
        
        # Simple rule-based validation to save API calls
        is_valid = has_content and has_structure and source_count > 0
        
        if is_valid:
            score = min(10, 5 + (word_count // 100) + (source_count // 2))
        else:
            score = 3
        
        return {
            'is_valid': is_valid,
            'score': score,
            'issues': [] if is_valid else ['Insufficient content or structure'],
            'metrics': {
                'word_count': word_count,
                'source_count': source_count,
                'has_structure': has_structure
            }
        }
    


    #support me...