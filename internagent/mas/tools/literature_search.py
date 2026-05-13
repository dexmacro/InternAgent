"""
Literature Search Tool for InternAgent

Rewritten version - Uses multiple reliable free academic search APIs
Supported sources: arXiv, Semantic Scholar, CrossRef, CORE
"""

import os
import asyncio
import logging
import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import urllib.parse
import requests

try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "aiohttp"])
    import aiohttp

try:
    from bs4 import BeautifulSoup
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "beautifulsoup4", "lxml"])
    from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Paper metadata"""
    
    title: str
    authors: List[str]
    abstract: str = ""
    content: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    citations: Optional[int] = None
    pdf_url: Optional[str] = None
    source: str = "unknown"  # Source: arxiv, semantic_scholar, crossref, core
    paper_id: Optional[str] = None  # Semantic Scholar paper ID
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_citation(self, format_type: str = "apa") -> str:
        """Generate citation format"""
        if format_type == "bibtex":
            first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
            year = self.year or "Unknown"
            key = f"{first_author}{year}"
            authors = " and ".join(self.authors) if self.authors else "Unknown"
            
            return (
                f"@article{{{key},\n"
                f"  title = {{{self.title}}},\n"
                f"  author = {{{authors}}},\n"
                f"  year = {{{year}}},\n"
                f"  journal = {{{self.journal or 'Unknown'}}},\n"
                f"  doi = {{{self.doi or ''}}},\n"
                f"  url = {{{self.url or ''}}}\n"
                f"}}"
            )
        
        # APA format (default)
        authors_str = ""
        if self.authors:
            if len(self.authors) == 1:
                authors_str = self.authors[0]
            elif len(self.authors) == 2:
                authors_str = f"{self.authors[0]} & {self.authors[1]}"
            else:
                authors_str = f"{self.authors[0]} et al."
        
        year_str = f"({self.year})" if self.year else ""
        journal_str = f"{self.journal}." if self.journal else ""
        doi_str = f"https://doi.org/{self.doi}" if self.doi else ""
        
        return f"{authors_str} {year_str}. {self.title}. {journal_str} {doi_str}".strip()


class CitationManager:
    """Citation manager"""
    
    def __init__(self):
        self.papers: Dict[str, PaperMetadata] = {}
        
    def add_paper(self, paper: PaperMetadata) -> None:
        """Add paper"""
        key = paper.doi if paper.doi else paper.title.lower().strip()
        if key not in self.papers:
            self.papers[key] = paper
    
    def clear(self) -> None:
        """Clear all papers"""
        self.papers.clear()

class LiteratureSearch:
    """
    Multi-source academic literature search tool
    
    Supported sources:
    - arXiv: Preprints in physics, mathematics, computer science, etc.
    - Semantic Scholar: Comprehensive academic search engine
    - CrossRef: DOI lookup and metadata
    - CORE: Open access research papers
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize literature search tool
        
        Args:
            email: Email for API access (optional)
            api_keys: API key dictionary (optional)
            citation_manager: Citation manager
            timeout: Request timeout in seconds
        """
        self.email = config.get("email") or "user@example.com"
        self.api_keys = config.get("api_keys") or {}
        self.timeout = config.get("timeout") or 30
        
        self.kg_config = config.get('kg_papers', {}) if config else {}
        
        self.citation_manager = CitationManager()
        # Cache
        self._cache: Dict[str, List[PaperMetadata]] = {}
        
        # Default configuration
        self.default_max_results = 10
        
        # User-Agent for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        }
    
    async def search_arxiv(self,
                          query: str,
                          max_results: int = 10,
                          **kwargs) -> List[PaperMetadata]:
        """
        Search arXiv preprints
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        cache_key = f"arxiv:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"[arXiv] Using cached results: {query}")
            return self._cache[cache_key]
        
        logger.debug(f"[arXiv] Searching: {query}")
        
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=self.headers) as response:
                    if response.status != 200:
                        logger.error(f"[arXiv] Request failed: {response.status}")
                        return []
                    
                    xml_text = await response.text()
                    papers = self._parse_arxiv_xml(xml_text)
                    
                    # logger.info(f"[arXiv] Found {len(papers)} papers")
                    
                    self._cache[cache_key] = papers
                    for paper in papers:
                        self.citation_manager.add_paper(paper)
                    
                    return papers
                    
        except asyncio.TimeoutError:
            logger.error(f"[arXiv] Request timeout")
            return []
        except Exception as e:
            logger.error(f"[arXiv] Search error: {e}")
            return []
    
    async def search_semantic_scholar(self,
                                      query: str,
                                      max_results: int = 10,
                                      **kwargs) -> List[PaperMetadata]:
        """
        Search Semantic Scholar
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        cache_key = f"semantic:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"[Semantic Scholar] Using cached results: {query}")
            return self._cache[cache_key]
        
        logger.debug(f"[Semantic Scholar] Searching: {query}")
        
        # API configuration
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        api_key = os.getenv("S2_API_KEY") or self.api_keys.get("semantic_scholar")
        
        headers = dict(self.headers)
        if api_key:
            headers["x-api-key"] = api_key
        
        params = {
            "query": query,
            "limit": min(max_results, 100),  # API limit
            "fields": "paperId,title,abstract,authors,year,venue,url,citationCount,externalIds,openAccessPdf"
        }
        
        try:
            await asyncio.sleep(1)  # Rate limiting
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 429:
                        logger.warning(f"[Semantic Scholar] Rate limited, waiting to retry...")
                        await asyncio.sleep(5)
                        return []
                    
                    if response.status != 200:
                        logger.error(f"[Semantic Scholar] Request failed: {response.status}")
                        return []
                    
                    data = await response.json()
                    papers = []
                    
                    for item in data.get("data", []):
                        authors = [a.get("name", "") for a in item.get("authors", [])]
                        external_ids = item.get("externalIds", {})
                        pdf_info = item.get("openAccessPdf", {})
                        
                        paper = PaperMetadata(
                            title=item.get("title", ""),
                            authors=authors,
                            abstract=item.get("abstract", ""),
                            year=item.get("year"),
                            doi=external_ids.get("DOI"),
                            journal=item.get("venue"),
                            url=item.get("url"),
                            citations=item.get("citationCount"),
                            pdf_url=pdf_info.get("url") if pdf_info else None,
                            source="semantic_scholar",
                            paper_id=item.get("paperId")
                        )
                        papers.append(paper)
                    
                    # logger.info(f"[Semantic Scholar] Found {len(papers)} papers")
                    
                    self._cache[cache_key] = papers
                    for paper in papers:
                        self.citation_manager.add_paper(paper)
                    
                    return papers
                    
        except asyncio.TimeoutError:
            logger.error(f"[Semantic Scholar] Request timeout")
            return []
        except Exception as e:
            logger.error(f"[Semantic Scholar] Search error: {e}")
            return []
    
    async def search_crossref(self,
                             query: str,
                             max_results: int = 10,
                             **kwargs) -> List[PaperMetadata]:
        """
        Search CrossRef (DOI database)
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        cache_key = f"crossref:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"[CrossRef] Using cached results: {query}")
            return self._cache[cache_key]
        
        logger.debug(f"[CrossRef] Searching: {query}")
        
        url = "https://api.crossref.org/works"
        params = {
            "query": query,
            "rows": max_results,
            "select": "DOI,title,author,abstract,published,container-title,URL"
        }
        
        headers = dict(self.headers)
        headers["mailto"] = self.email
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"[CrossRef] Request failed: {response.status}")
                        return []
                    
                    data = await response.json()
                    papers = []
                    
                    for item in data.get("message", {}).get("items", []):
                        # Extract authors
                        authors = []
                        for author in item.get("author", []):
                            given = author.get("given", "")
                            family = author.get("family", "")
                            if given and family:
                                authors.append(f"{given} {family}")
                            elif family:
                                authors.append(family)
                        
                        # Extract title
                        title_list = item.get("title", [])
                        title = title_list[0] if title_list else ""
                        
                        # Extract year
                        year = None
                        pub_date = item.get("published", {}).get("date-parts", [[]])[0]
                        if pub_date:
                            year = pub_date[0] if len(pub_date) > 0 else None
                        
                        # Extract journal
                        journal_list = item.get("container-title", [])
                        journal = journal_list[0] if journal_list else None
                        
                        paper = PaperMetadata(
                            title=title,
                            authors=authors,
                            abstract=item.get("abstract", ""),
                            year=year,
                            doi=item.get("DOI"),
                            journal=journal,
                            url=item.get("URL"),
                            source="crossref"
                        )
                        papers.append(paper)
                    
                    # logger.info(f"[CrossRef] Found {len(papers)} papers")
                    
                    self._cache[cache_key] = papers
                    for paper in papers:
                        self.citation_manager.add_paper(paper)
                    
                    return papers
                    
        except asyncio.TimeoutError:
            logger.error(f"[CrossRef] Request timeout")
            return []
        except Exception as e:
            logger.error(f"[CrossRef] Search error: {e}")
            return []
    
    async def search_core(self,
                         query: str,
                         max_results: int = 10,
                         **kwargs) -> List[PaperMetadata]:
        """
        Search CORE (open access papers)
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata
        """
        cache_key = f"core:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"[CORE] Using cached results: {query}")
            return self._cache[cache_key]
        
        logger.debug(f"[CORE] Searching: {query}")
        
        api_key = os.getenv("CORE_API_KEY") or self.api_keys.get("core")
        if not api_key:
            logger.warning("[CORE] No API key available, skipping search")
            return []
        
        url = "https://api.core.ac.uk/v3/search/works"
        headers = dict(self.headers)
        headers["Authorization"] = f"Bearer {api_key}"
        
        params = {
            "q": query,
            "limit": max_results
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"[CORE] Request failed: {response.status}")
                        return []
                    
                    data = await response.json()
                    papers = []
                    
                    for item in data.get("results", []):
                        authors = [a.get("name", "") for a in item.get("authors", [])]
                        
                        paper = PaperMetadata(
                            title=item.get("title", ""),
                            authors=authors,
                            abstract=item.get("abstract", ""),
                            year=item.get("yearPublished"),
                            doi=item.get("doi"),
                            journal=item.get("publisher"),
                            url=item.get("downloadUrl"),
                            pdf_url=item.get("downloadUrl"),
                            source="core"
                        )
                        papers.append(paper)
                    
                    # logger.info(f"[CORE] Found {len(papers)} papers")
                    
                    self._cache[cache_key] = papers
                    for paper in papers:
                        self.citation_manager.add_paper(paper)
                    
                    return papers
                    
        except asyncio.TimeoutError:
            logger.error(f"[CORE] Request timeout")
            return []
        except Exception as e:
            logger.error(f"[CORE] Search error: {e}")
            return []
        
    async def search_kg_papers(self,
                         query: str,
                         max_results: Optional[int] = None,
                         **kwargs) -> List[PaperMetadata]: 
        """
        Search local knowledge graph paper database using the retrieval API.
        
        Args:
            query: Search query string
            max_results: Number of top results to return (default: from config or 3)
        
        Returns:
            List of paper dictionaries with metadata and content
            
        Example:
            >>> papers = search_kg_papers("machine learning", top_k=5)
            >>> for paper in papers:
            >>>     print(paper['title'])
            >>>     print(paper.get('content', 'No content available'))
            >>>     print(f"Is stub: {paper.get('is_stub', False)}")
        """
        # Load configuration
        cache_key = f"kg:{query}:{max_results}"
        if cache_key in self._cache:
            logger.info(f"[CrossRef] Using cached results: {query}")
            return self._cache[cache_key]
        # Use provided values or fall back to config or defaults
        
        api_url = self.kg_config.get('api_url', 'localhost:5001')
        max_results = max_results

        search_endpoint = f"{api_url}/search_kg_papers"

        payload = {
            "query": query,
            "top_k": max_results,
        }
        
        try:
            logger.debug(f"[Knowledge Graph]Searching: '{query}' (top_k={max_results})")
            
            response = requests.post(
                search_endpoint,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=300
            )

            if response.status_code != 200:
                logger.error(f"Search failed with status {response.status_code}: {response.text}")
                return []
            
            data = response.json()
            
            papers = []
            for paper in data:
                paper = PaperMetadata(
                    title=paper.get('title', ''),
                    authors=paper.get('authors', []),
                    abstract=paper.get('abstract', ''),
                    content=paper.get('content', ''),
                    year=paper.get('year', None),
                    doi=paper.get('doi', None),
                    url=paper.get('url', None),
                    citations=paper.get('citations', 0),
                    pdf_url=paper.get('pdf_url', None),
                    source="kg_papers"
                )
                papers.append(paper)            
            
            if not papers:
                logger.info(f"No results found for query: '{query}'")
                return []
            
            # logger.info(f"[Knowledge Graph] Found {len(papers)} papers from knowledge graph")
            self._cache[cache_key] = papers
            
            for paper in papers:
                self.citation_manager.add_paper(paper)
            
            return papers
            
        except requests.exceptions.Timeout:
            logger.error(f"KG Search request timed out after 300 seconds")
            return []
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to KG API at {api_url}. Is the server running?")
            return []
        except Exception as e:
            logger.error(f"Error searching KG papers: {str(e)}")
            return []

    async def multi_source_search(self,
                                  query: str,
                                  sources: Optional[List[str]] = None,
                                  max_results: int = 10,
                                  **kwargs) -> Dict[str, List[PaperMetadata]]:
        """
        Multi-source parallel search
        
        Args:
            query: Search query
            sources: List of sources (arxiv, semantic_scholar, crossref, core)
            max_results: Maximum number of results per source
            
        Returns:
            Mapping from source name to result list
        """
        
        tasks = []
        task_sources = []
        
        for source in sources:
            if source == "arxiv":
                tasks.append(self.search_arxiv(query, max_results, **kwargs))
                task_sources.append("arxiv")
            elif source == "semantic_scholar":
                tasks.append(self.search_semantic_scholar(query, max_results, **kwargs))
                task_sources.append("semantic_scholar")
            elif source == "crossref":
                tasks.append(self.search_crossref(query, max_results, **kwargs))
                task_sources.append("crossref")
            elif source == "core":
                tasks.append(self.search_core(query, max_results, **kwargs))
                task_sources.append("core")
            elif source == "kg_papers":
                tasks.append(self.search_kg_papers(query, max_results, **kwargs))
                task_sources.append("kg_papers")    
        
        logger.debug(f"[Multi-source search] Executing searches on sources: {task_sources} for query {query}" )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined = {}
        for source, result in zip(task_sources, results):
            if isinstance(result, Exception):
                logger.error(f"[{source}] Error: {result}")
                combined[source] = []
            else:
                combined[source] = result
        
        total = sum(len(papers) for papers in combined.values())
        logger.info(f"[Multi-source search] Found {total} papers in total")
        
        return combined
    
    async def search(self,
                    query: str,
                    max_results: int = 10,
                    sources: Optional[List[str]] = None,
                    **kwargs) -> List[PaperMetadata]:
        """
        Simplified search interface - merges results from multiple sources
        
        Args:
            query: Search query
            max_results: Total maximum number of results
            sources: List of sources
            
        Returns:
            Merged and deduplicated list of papers
        """
        results_dict = await self.multi_source_search(
            query, 
            sources=sources, 
            max_results=max_results,
            **kwargs
        )
        
        # Merge and deduplicate
        seen_titles = set()
        all_papers = []
        
        for source, papers in results_dict.items():
            for paper in papers:
                title_key = paper.title.lower().strip()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_papers.append(paper)
        
        return all_papers[:max_results]
    
    def clear_cache(self) -> None:
        """Clear search cache"""
        self._cache.clear()
        logger.info("Cache cleared")
    
    # ------------------------------------------------------------------
    # Citation-graph helpers
    # ------------------------------------------------------------------

    def _make_s2_headers(self) -> Dict[str, str]:
        """Build headers for Semantic Scholar API requests."""
        headers = dict(self.headers)
        api_key = os.getenv("S2_API_KEY") or self.api_keys.get("semantic_scholar")
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    async def resolve_paper_id(self, identifier: str) -> Optional[str]:
        """
        Resolve a paper identifier (title, DOI, or arXiv ID) to a Semantic Scholar
        paper ID.

        The method probes the identifier in this order:
        1. If it looks like an arXiv ID (e.g. ``2301.01234`` or ``cs.LG/0612046``)
           it queries ``/paper/ARXIV:{id}``.
        2. If it looks like a DOI (contains ``/`` but is not clearly a title) it
           queries ``/paper/DOI:{doi}``.
        3. Otherwise it performs a full-text search and returns the top hit's ID.

        Args:
            identifier: Paper title, DOI, or arXiv ID string.

        Returns:
            Semantic Scholar ``paperId`` string, or ``None`` if not found.
        """
        cache_key = f"resolve:{identifier}"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        headers = self._make_s2_headers()
        base = "https://api.semanticscholar.org/graph/v1/paper"
        fields = "paperId,title,externalIds"

        async def _fetch_paper(pid: str) -> Optional[str]:
            url = f"{base}/{pid}"
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        url, params={"fields": fields}, headers=headers
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("paperId")
            except Exception as e:
                logger.debug(f"[resolve_paper_id] Probe {pid} failed: {e}")
            return None

        paper_id: Optional[str] = None

        # 1. ArXiv ID heuristic: digits with a dot, or old-style category/YYMM.NNNNN
        arxiv_pattern = re.compile(r"^(\d{4}\.\d{4,5}(v\d+)?|[a-zA-Z\-]+/\d{7})$")
        if arxiv_pattern.match(identifier.strip()):
            paper_id = await _fetch_paper(f"ARXIV:{identifier.strip()}")

        # 2. DOI heuristic: contains "/" but does not look like a sentence
        if paper_id is None and "/" in identifier and " " not in identifier:
            paper_id = await _fetch_paper(f"DOI:{identifier.strip()}")

        # 3. Title / keyword search fallback
        if paper_id is None:
            try:
                await asyncio.sleep(1)
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        f"{base}/search",
                        params={"query": identifier, "limit": 1, "fields": fields},
                        headers=headers,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            hits = data.get("data", [])
                            if hits:
                                paper_id = hits[0].get("paperId")
            except Exception as e:
                logger.error(f"[resolve_paper_id] Search fallback failed: {e}")

        self._cache[cache_key] = paper_id  # type: ignore[assignment]
        return paper_id

    async def get_references_by_paper_id(
        self, paper_id: str, max_results: int = 50
    ) -> List[PaperMetadata]:
        """
        Fetch the reference list of a paper via the Semantic Scholar API.

        Calls ``GET /paper/{paper_id}/references`` and returns the papers that
        *this* paper cites (i.e. its bibliography).

        Args:
            paper_id: Semantic Scholar ``paperId`` (or ``ARXIV:…`` / ``DOI:…``
                      prefixed identifier).
            max_results: Maximum number of references to return.

        Returns:
            List of :class:`PaperMetadata` objects for each referenced paper.
        """
        cache_key = f"refs:{paper_id}:{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
        )
        fields = (
            "paperId,title,abstract,authors,year,venue,url,"
            "citationCount,externalIds,openAccessPdf"
        )
        params: Dict[str, Any] = {"fields": fields, "limit": min(max_results, 1000)}
        headers = self._make_s2_headers()

        papers: List[PaperMetadata] = []
        try:
            await asyncio.sleep(1)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 429:
                        logger.warning("[S2 references] Rate limited")
                        await asyncio.sleep(5)
                        return []
                    if resp.status != 200:
                        logger.error(
                            f"[S2 references] Request failed: {resp.status}"
                        )
                        return []
                    data = await resp.json()
                    for entry in data.get("data", []):
                        cited = entry.get("citedPaper", {})
                        if not cited or not cited.get("title"):
                            continue
                        authors = [
                            a.get("name", "") for a in cited.get("authors", [])
                        ]
                        ext_ids = cited.get("externalIds") or {}
                        pdf_info = cited.get("openAccessPdf") or {}
                        paper = PaperMetadata(
                            title=cited.get("title", ""),
                            authors=authors,
                            abstract=cited.get("abstract") or "",
                            year=cited.get("year"),
                            doi=ext_ids.get("DOI"),
                            journal=cited.get("venue"),
                            url=cited.get("url"),
                            citations=cited.get("citationCount"),
                            pdf_url=pdf_info.get("url"),
                            source="semantic_scholar",
                            paper_id=cited.get("paperId"),
                        )
                        papers.append(paper)
                        self.citation_manager.add_paper(paper)
        except asyncio.TimeoutError:
            logger.error("[S2 references] Request timeout")
        except Exception as e:
            logger.error(f"[S2 references] Error: {e}")

        self._cache[cache_key] = papers
        logger.info(
            f"[S2 references] Found {len(papers)} references for {paper_id}"
        )
        return papers

    async def get_citing_papers(
        self, paper_id: str, max_results: int = 50
    ) -> List[PaperMetadata]:
        """
        Fetch papers that cite the given paper via the Semantic Scholar API.

        Calls ``GET /paper/{paper_id}/citations`` and returns papers that *cite*
        this paper (i.e. papers that appear in other works' reference lists).

        Args:
            paper_id: Semantic Scholar ``paperId`` (or ``ARXIV:…`` / ``DOI:…``
                      prefixed identifier).
            max_results: Maximum number of citing papers to return.

        Returns:
            List of :class:`PaperMetadata` objects for each citing paper.
        """
        cache_key = f"cites:{paper_id}:{max_results}"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
        )
        fields = (
            "paperId,title,abstract,authors,year,venue,url,"
            "citationCount,externalIds,openAccessPdf"
        )
        params: Dict[str, Any] = {"fields": fields, "limit": min(max_results, 1000)}
        headers = self._make_s2_headers()

        papers: List[PaperMetadata] = []
        try:
            await asyncio.sleep(1)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 429:
                        logger.warning("[S2 citations] Rate limited")
                        await asyncio.sleep(5)
                        return []
                    if resp.status != 200:
                        logger.error(
                            f"[S2 citations] Request failed: {resp.status}"
                        )
                        return []
                    data = await resp.json()
                    for entry in data.get("data", []):
                        citing = entry.get("citingPaper", {})
                        if not citing or not citing.get("title"):
                            continue
                        authors = [
                            a.get("name", "") for a in citing.get("authors", [])
                        ]
                        ext_ids = citing.get("externalIds") or {}
                        pdf_info = citing.get("openAccessPdf") or {}
                        paper = PaperMetadata(
                            title=citing.get("title", ""),
                            authors=authors,
                            abstract=citing.get("abstract") or "",
                            year=citing.get("year"),
                            doi=ext_ids.get("DOI"),
                            journal=citing.get("venue"),
                            url=citing.get("url"),
                            citations=citing.get("citationCount"),
                            pdf_url=pdf_info.get("url"),
                            source="semantic_scholar",
                            paper_id=citing.get("paperId"),
                        )
                        papers.append(paper)
                        self.citation_manager.add_paper(paper)
        except asyncio.TimeoutError:
            logger.error("[S2 citations] Request timeout")
        except Exception as e:
            logger.error(f"[S2 citations] Error: {e}")

        self._cache[cache_key] = papers
        logger.info(
            f"[S2 citations] Found {len(papers)} citing papers for {paper_id}"
        )
        return papers

    def _parse_arxiv_xml(self, xml_data: str) -> List[PaperMetadata]:
        """
        Parse arXiv XML response
        
        Args:
            xml_data: arXiv XML response
            
        Returns:
            List of paper metadata
        """
        papers = []
        
        try:
            soup = BeautifulSoup(xml_data, "xml")
            
            for entry in soup.find_all("entry"):
                try:
                    # Title
                    title_elem = entry.find("title")
                    title = title_elem.text.strip().replace("\n", " ") if title_elem else ""
                    
                    # Abstract
                    summary_elem = entry.find("summary")
                    abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem else ""
                    
                    # Authors
                    authors = []
                    for author in entry.find_all("author"):
                        name_elem = author.find("name")
                        if name_elem:
                            authors.append(name_elem.text.strip())
                    
                    # Publication year
                    year = None
                    published_elem = entry.find("published")
                    if published_elem:
                        match = re.search(r"(\d{4})", published_elem.text)
                        if match:
                            year = int(match.group(1))
                    
                    # URL and PDF
                    url = None
                    pdf_url = None
                    for link in entry.find_all("link"):
                        href = link.get("href", "")
                        if link.get("rel") == "alternate":
                            url = href
                        elif link.get("title") == "pdf":
                            pdf_url = href
                    
                    # arXiv ID
                    id_elem = entry.find("id")
                    if id_elem and not url:
                        url = id_elem.text.strip()
                    
                    if not pdf_url and url:
                        # Generate PDF link from URL
                        pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
                    
                    if title:
                        paper = PaperMetadata(
                            title=title,
                            authors=authors,
                            abstract=abstract,
                            year=year,
                            journal="arXiv",
                            url=url,
                            pdf_url=pdf_url,
                            source="arxiv"
                        )
                        papers.append(paper)
                        
                except Exception as e:
                    logger.debug(f"Error parsing single arXiv entry: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing arXiv XML: {e}")
        
        return papers
