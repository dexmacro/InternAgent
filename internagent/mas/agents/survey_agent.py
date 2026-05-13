"""
Survey Agent for InternAgent

This module implements the Survey Agent, which performs comprehensive literature
surveys on research topics. The agent generates intelligent search queries, retrieves
relevant academic papers from multiple sources, scores papers based on relevance,
and performs deep reading analysis to extract methodological details from top papers.
This agent supports automated, iterative literature review with query refinement.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import asyncio
from .base_agent import BaseAgent, AgentExecutionError
from ..tools.literature_search import LiteratureSearch, PaperMetadata
from ..tools.citation_graph import CitationGraph
from ..tools.web_search import WebSearch
from ..tools.utils import parse_io_description, format_papers_for_printing_next_query,\
    download_pdf, extract_text_from_pdf, download_pdf_by_doi, select_papers

logger = logging.getLogger(__name__)

# Maximum number of characters from a paper's text fed into the deep-read LLM
# prompt.  Keeps prompts within typical context-window budgets while covering
# the most information-dense sections of a paper.
_DEEP_READ_MAX_CHARS = 8000


class SurveyAgent(BaseAgent):
    """
    Survey Agent conducts comprehensive literature surveys for research topics.

    This agent performs intelligent literature search by:
    - Generating context-aware search queries based on research topics
    - Retrieving papers from multiple academic sources (Semantic Scholar, arXiv, CrossRef, CORE)
    - Iteratively refining search queries to expand paper coverage
    - Scoring papers based on relevance, novelty, and methodological quality
    - Performing deep reading analysis on top-ranked papers to extract methodological details

    The agent employs an iterative search strategy that starts with keyword queries
    and progressively diversifies using paper similarity and reference-based queries
    to build a comprehensive literature bank.
    
    Supported Search Tools:
    - literature_search: Searches academic papers from arXiv, Semantic Scholar, CrossRef, CORE, KG Papers
    - web_search: Searches web pages (general web content) via Google Serper API
    
    Note: literature_search is for academic papers, web_search is for general web content.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the survey agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.max_papers = config.get("max_papers", 5) # max paper for download and deep read
        self.search_depth = config.get("search_depth", "moderate")  # shallow, moderate, deep
        self.sources = config.get("sources", ["arxiv", "crossref"])
        self.max_concurrent_tasks = 10
        # Initialize tools
        tools_config = config.get("_global_config", {}).get("tools", {})
        self.literature_search = None
        self.web_search = None
        
        self._init_literature_search(tools_config.get("literature_search", {}))
        self._init_web_search(tools_config.get("web_search", {}))
    
    def _init_literature_search(self, config: Dict[str, Any]) -> None:
        """
        Initialize the literature search tool (multi-source academic search).
        
        Args:
            config: Literature search configuration
        """
        try:
            self.literature_search = LiteratureSearch(
                config=config
            )
            logger.info("Literature search tool initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize literature search: {str(e)}")
    
    def _init_web_search(self, config: Dict[str, Any]) -> None:
        """
        Initialize the web search tool.
        """
        try:
            self.web_search = WebSearch(config=config)
            logger.info("Web search tool initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize web search: {str(e)}")
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute survey agent with configurable search sources.
        
        When ``context`` contains a ``seed_paper`` key the agent switches into
        *citation-graph mode* and calls :meth:`build_citation_graph` instead of
        the regular keyword-based survey.

        Args:
            context: Context dictionary containing research topic information.
                     Set ``seed_paper`` (title / DOI / arXiv ID) to trigger
                     citation-graph mode.
            params: Parameters including optional 'search_sources' to specify
                    which search tools to use.
                    Available sources: 'literature_search', 'web_search'
        
        Returns:
            Dict containing:
                - 'papers': List of academic papers (from literature_search)
                - 'web_results': List of web pages (from web_search)
            Or in citation-graph mode:
                - 'citation_graph': serialised graph dict (nodes + edges)
                - 'papers': flat list of all paper dicts in the graph
                - 'graph_json_path': path to the saved JSON file (if any)
                - 'graph_dot_path': path to the saved DOT file (if any)
                - 'graph_png_path': path to the saved PNG file (if any)
        """
        # --- Citation-graph mode ---
        if context.get("seed_paper"):
            return await self.build_citation_graph(context, params)

        results = {
            "papers": [],
            "web_results": []
        }
        
        # Execute literature search for academic papers
            
        # Execute web search for web pages (separate from papers)
        if 'web_search' in self.sources:
            web_results, _ = await self.web_search_query(context=context)
            results["web_results"] = web_results
            # remove 'web_search' from sources to avoid duplication
            self.sources = [src for src in self.sources if src != 'web_search']
            
        papers, _ = await self.advanced_query_paper(context=context)
        results["papers"] = papers
        
        return results
    
    async def literature_search_query(self, 
                                     query: str, 
                                     max_results: int = 10) -> Dict[str, Any]:
        """
        Search academic literature only (NOT web pages).
        
        Args:
            query: Search query string
            max_results: Maximum number of results per source
        
        Returns:
            Dict with source names as keys and paper lists as values
            Example: {'arxiv': [...], 'semantic_scholar': [...], 'kg_papers': [...]}
        """
        all_results = {}
        
        try:
            logger.info(f"[Literature Search] Searching academic papers: {query} from sources {self.sources}")
            # Use configured sources or defaults (exclude kg_papers from multi_source_search)

            if self.sources:
                lit_results = await self.literature_search.multi_source_search(
                    query=query,
                    sources=self.sources,
                    max_results=max_results
                )
                
                # Convert PaperMetadata objects to dict format
                for source, papers in lit_results.items():
                    if papers:
                        formatted_papers = []
                        for paper in papers:
                            paper_dict = {
                                'title': paper.title,
                                'authors': paper.authors,
                                'abstract': paper.abstract,
                                'content': paper.content or '',
                                'year': paper.year,
                                'doi': paper.doi,
                                'url': paper.url,
                                'source': paper.source,
                                'citations': paper.citations,
                                'pdf_url': paper.pdf_url
                            }
                            formatted_papers.append(paper_dict)
                        
                        all_results[source] = formatted_papers
                    
        except Exception as e:
            logger.error(f"[Literature Search] Search error: {e}")
        
        return all_results
    
    async def web_search_query(self, context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Perform web search based on research context (NOT academic papers).
        
        Args:
            context: Context containing research topic information
        
        Returns:
            Tuple of (web_results, search_queries)
        """
        web_results = []
        search_queries = []
        
        if not self.web_search:
            logger.warning("Web search tool not initialized")
            return web_results, search_queries
        
        goal_description = context.get("description", "")
        
        try:
            raw_results = self.web_search.search_serper(goal_description, 10)
            
            if raw_results:
                for result in raw_results:
                    if 'error' not in result:
                        web_results.append({
                            'title': result.get('title', ''),
                            'description': result.get('long_description', ''),
                            'url': result.get('url', ''),
                            'source': 'web_search'
                        })
            logger.info(f"[Web search] found {len(web_results)} results")
        except Exception as e:
            logger.error(f"Web search error: {e}")
            raise AgentExecutionError(f"Failed to perform web search: {str(e)}")
        
        return web_results, search_queries


    async def advanced_query_paper(self, context) -> Tuple[List[Dict[str, Any]], List[str]]:
        # ------------------------------------------------------------
        # Helper: scoring single batch
        # ------------------------------------------------------------
        async def score_batch(batch_index, batch, semaphore):
            async with semaphore:
                # Prepare papers (content fallback to abstract)
                abs_batch = [
                    {
                        'id': paper['id'],
                        'title': paper['title'],
                        'content': paper['content'] if paper['content'] else paper['abstract']
                    }
                    for paper in batch
                ]

                # Build prompt
                if io_description is not None:
                    prompt = (
                        "You are a precise and reliable literature-review scoring assistant.\n\n"
                        "Read each paper in the list below and assign a score to each one individually.\n"
                        "Each paper is a dictionary containing: id, title, and content.\n\n"
                        "Scoring criteria:\n"
                        f"1. Relevance to the domain: {domain}\n"
                        f"2. Input/Output match:\n"
                        f"   - Input: {io_description[0]}\n"
                        f"   - Output: {io_description[1]}\n"
                        "3. Empirical novelty of the method\n"
                        "4. Interestingness and meaningfulness\n\n"
                        "Instructions:\n"
                        "- Score each paper independently.\n"
                        "- Use a scoring scale from 1 to 10 (10 = excellent match).\n"
                        "- Do NOT add new papers. Do NOT modify IDs.\n"
                        "- Return only a JSON object where:\n"
                        "    keys = paper.id\n"
                        "    values = numeric scores\n\n"
                        f"The papers to score are:\n{abs_batch}\n\n"
                        "Return JSON only."
                    )
                else:
                    prompt = (
                        "You are a precise and reliable literature-review scoring assistant.\n\n"
                        "Read each paper in the list below and assign a score to each one individually.\n"
                        "Each paper is a dictionary containing: id, title, and content.\n\n"
                        "Scoring criteria:\n"
                        "1. Relevance to the target topic\n"
                        "2. Novelty\n"
                        "3. Empirical strength\n"
                        "4. Meaningfulness\n\n"
                        "Instructions:\n"
                        "- Score each paper independently.\n"
                        "- Use a scoring scale from 1 to 10.\n"
                        "- Do NOT add new papers. Do NOT modify IDs.\n"
                        "- Return only a JSON object where:\n"
                        "    keys = paper.id\n"
                        "    values = numeric scores\n\n"
                        f"The papers to score are:\n{abs_batch}\n\n"
                        "Return JSON only."
                    )

                # Call model
                try:
                    response = await self._call_model(
                        prompt=prompt,
                        schema=output_schema_paper_score
                    )
                    return batch_index, response
                except Exception as e:
                    logger.error(f"Failed scoring batch {batch_index}: {e}")
                    return batch_index, {}


        # ------------------------------------------------------------
        # Helper: literature search single query
        # ------------------------------------------------------------
        async def run_lit_query(q, semaphore):
            async with semaphore:
                try:
                    out = await self.literature_search_query(q, 10)
                    return q, out
                except Exception as e:
                    logger.error(f"Query {q} failed: {e}")
                    return q, None

        # ------------------------------------------------------------
        # Extract context
        # ------------------------------------------------------------
        search_queries = []
        goal_description = context.get("description", {})
        domain = context.get("domain", "")

        # Schemas
        output_schema_paper_score = {
            "type": "object",
            "Properties": {
                "^[a-zA-Z0-9_]+$": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 10
                }
            }
        }

        output_schema_paper_details = {
            "type": "object",
            "properties": {
                "background": {"type": "string"},
                "contributions": {"type": "string"},
                "methods": {"type": "string"},
                "challenges": {"type": "string"}
            },
            "required": ["background", "contributions", "methods", "challenges"]
        }

        # ------------------------------------------------------------
        # Step 1: define task attribute
        # ------------------------------------------------------------
        define_task_attribute_prompt = (
            f"You are a researcher working on: {domain}. "
            f"Define task attributes using Input(...) Output(...). "
            f"Just return the attribute itself."
        )
        try:
            response = await self._call_model(prompt=define_task_attribute_prompt)
            io_description = parse_io_description(response)
        except Exception as e:
            logger.error(f"Error defining task attribute: {e}")
            raise AgentExecutionError("Failed to define task attribute")

        # ------------------------------------------------------------
        # Step 2: initial query (KeywordQuery)
        # ------------------------------------------------------------
        init_keyword_query_prompt = (
            f"You are a researcher doing literature review on {goal_description}. "
            f"Propose keywords for Semantic Scholar. "
            f"Return KeywordQuery('...') only."
        )
        try:
            response = await self._call_model(prompt=init_keyword_query_prompt)
            init_query = response
        except Exception as e:
            logger.error("Initial keyword query generation failed", exc_info=e)
            raise AgentExecutionError("Failed to generate initial keyword query")

        # Run initial query
        init_paper_lst = await self.literature_search_query(init_query, 10)
        search_queries.append(init_query)

        # Flatten initial papers
        if init_paper_lst:
            flattened = []
            for src, papers in init_paper_lst.items():
                if isinstance(papers, list):
                    flattened.extend(papers)
                elif isinstance(papers, dict) and "data" in papers:
                    flattened.extend(papers["data"])
            paper_bank = {str(i): p for i, p in enumerate(flattened)}
        else:
            paper_bank = {}

        # ------------------------------------------------------------
        # Step 3: generate remaining queries (串行生成，但暂不执行)
        # ------------------------------------------------------------
        all_queries = []
        iteration = 0

        while len(paper_bank) < self.max_papers and iteration < 10:
            data_list = [
                {'id': id, **info}
                for id, info in paper_bank.items()
            ]
            grounding_k = 10
            grounding_papers = data_list[:grounding_k]
            grounding_str = format_papers_for_printing_next_query(grounding_papers)

            if io_description:
                new_query_prompt = (
                    f"You are a researcher studying {domain}. "
                    f"Generate a new query (PaperQuery / KeywordQuery / GetReferences) "
                    f"based on current results. "
                    f"Input={io_description[0]} Output={io_description[1]}. "
                    f"Papers so far: {grounding_str}. "
                    f"Previous queries: {search_queries}. "
                    f"Return ONLY the new query."
                )
            else:
                new_query_prompt = (
                    f"You are a researcher studying {domain}. "
                    f"Generate new query based on: {grounding_str}. "
                    f"Previous queries: {search_queries}. "
                    f"Return ONLY the query."
                )

            try:
                response = await self._call_model(prompt=new_query_prompt)
                new_query = response
                all_queries.append(new_query)
                search_queries.append(new_query)
            except Exception as e:
                logger.error(f"Error generating new query: {e}")
                break

            iteration += 1

        # ------------------------------------------------------------
        # Step 4: run all literature queries concurrently
        # ------------------------------------------------------------
        semaphore_lit = asyncio.Semaphore(self.max_concurrent_tasks)

        lit_tasks = [
            asyncio.create_task(run_lit_query(q, semaphore_lit))
            for q in all_queries
        ]
        lit_results = await asyncio.gather(*lit_tasks)

        # process results
        for q, new_paper_lst in lit_results:
            if not new_paper_lst:
                continue

            flattened = []
            for source, papers in new_paper_lst.items():
                if isinstance(papers, list):
                    flattened.extend(papers)
                elif isinstance(papers, dict) and "data" in papers:
                    flattened.extend(papers["data"])

            existing_titles = {p['title'] for p in paper_bank.values()}
            new_papers = [p for p in flattened if p['title'] not in existing_titles]

            if new_papers:
                start = len(paper_bank)
                for i, p in enumerate(new_papers):
                    paper_bank[str(start+i)] = p

        # ------------------------------------------------------------
        # Step 5: scoring (并发 + 写回修复)
        # ------------------------------------------------------------
        data_list = [{'id': id, **info} for id, info in paper_bank.items()]
        paper_bank = data_list[:]  # convert to list

        BATCH_SIZE = 10
        batches = []
        for batch_index in range(0, len(paper_bank), BATCH_SIZE):
            batch = paper_bank[batch_index:batch_index + BATCH_SIZE]
            batches.append((batch_index, batch))

        semaphore_score = asyncio.Semaphore(self.max_concurrent_tasks)

        score_tasks = [
            asyncio.create_task(score_batch(bi, batch, semaphore_score))
            for bi, batch in batches
        ]
        score_results = await asyncio.gather(*score_tasks)
        logger.info(f"Completed scoring all batches: {score_results}")
        # ------------------------------------------------------------
        # FIX: scoring 写回逻辑（绝不再出现 KeyError: 'score'）
        # ------------------------------------------------------------
        for batch_index, score_dict in score_results:
            batch_start = batch_index
            batch_end = min(batch_index + BATCH_SIZE, len(paper_bank))
            batch_size = batch_end - batch_start

            # 1. 全部先初始化默认 score（避免缺失值导致 KeyError）
            for global_id in range(batch_start, batch_end):
                paper_bank[global_id]['score'] = 1  # 默认最低分

            # 2. 用 LLM 返回结果覆盖
            if isinstance(score_dict, dict):
                for key, score in score_dict.items():
                    try:
                        local_id = int(key)
                        if 0 <= local_id < batch_size:
                            global_id = batch_start + local_id
                            paper_bank[global_id]['score'] = score
                    except:
                        continue

        logger.info(f"Number of papers in paper_bank: {len(paper_bank)}")

        # ------------------------------------------------------------
        # Step 6: deep read (与你原版本一致)
        # ------------------------------------------------------------
        rag_read_depth = 3
        selected_for_deep_read = select_papers(paper_bank, self.max_papers, rag_read_depth)

        base_dir = 'tmp'
        pdf_dir = os.path.join(base_dir, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)

        for paper in selected_for_deep_read:
            url = paper.get('url')
            doi = paper.get('doi')

            # -----------------------------------------------
            # 1. 优先使用已有的 paper.content，如果存在且非空
            # -----------------------------------------------
            content_text = paper.get("content")
            if isinstance(content_text, str) and content_text.strip():
                text = content_text.strip()
                logger.info(f"Using existing content from kg for paper ID {paper['id']}")
            else:
                # -----------------------------------------------
                # 2. 否则从 PDF 下载或通过 DOI 下载
                # -----------------------------------------------
                pdf_path = None
                if url:
                    pdf_path = download_pdf(url, save_folder=pdf_dir)
                if doi and not pdf_path:
                    pdf_path = download_pdf_by_doi(doi=doi, download_dir=pdf_dir)

                text = None
                if pdf_path:
                    text = extract_text_from_pdf(pdf_path)

            # -----------------------------------------------
            # 3. 只有当 text 存在时才进行 LLM 分析
            # -----------------------------------------------
            if text:
                detail_prompt = (
                    f"Analyze the following paper text: {text}\n"
                    f"Extract background, contributions, methods, challenges. Return JSON."
                )
                try:
                    response = await self._call_model(
                        prompt=detail_prompt,
                        schema=output_schema_paper_details
                    )
                    paper["background"] = response.get("background", "")
                    paper["contributions"] = response.get("contributions", "")
                    paper["methods"] = response.get("methods", "")
                    paper["challenges"] = response.get("challenges", "")
                except Exception:
                    pass

        selected_ids = [p['id'] for p in selected_for_deep_read]
        for p in paper_bank:
            p['is_deep_read'] = (p['id'] in selected_ids)

        return paper_bank, search_queries

    # ------------------------------------------------------------------
    # Citation-graph mode
    # ------------------------------------------------------------------

    async def build_citation_graph(
        self, context: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build a directed citation graph starting from a seed paper.

        Starting from the paper identified by ``context["seed_paper"]``, the
        method performs a BFS expansion up to ``context.get("depth", 2)``
        levels deep, collecting all referenced papers at each level.  Optionally
        it also collects papers that cite the seed paper when
        ``context.get("include_citing_papers", False)`` is ``True``.

        For every paper encountered the agent attempts a deep-read analysis
        (download PDF → extract text → call LLM) that populates the following
        structured fields on each graph node:

        - ``problem_and_background`` – the research problem the paper addresses
        - ``contributions`` – key contributions / solutions introduced
        - ``methods`` – technical approach
        - ``challenges`` – challenges faced / open problems
        - ``limitations_and_future_work`` – limitations and future directions

        Results are serialised to JSON (and optionally DOT / PNG) under the
        ``output_dir`` path (default ``tmp/citation_graph``).

        Args:
            context: Must contain ``seed_paper`` (str – title, DOI, or arXiv
                     ID).  Recognised optional keys:

                     - ``depth`` (int, default ``2``) – BFS depth limit
                     - ``max_refs_per_paper`` (int, default ``30``) – max
                       references to fetch per paper
                     - ``include_citing_papers`` (bool, default ``False``) –
                       also fetch papers that cite the seed
                     - ``output_dir`` (str, default ``"tmp/citation_graph"``) –
                       directory for output files
                     - ``save_dot`` (bool, default ``False``) – save DOT file
                     - ``save_png`` (bool, default ``False``) – save PNG file
            params: Unused; reserved for future extensions.

        Returns:
            Dict containing:
            - ``citation_graph`` – :meth:`CitationGraph.to_json` dict
            - ``papers`` – flat list of all paper dicts in the graph
            - ``graph_json_path`` – path to the JSON output file
            - ``graph_dot_path`` – path to DOT file or ``None``
            - ``graph_png_path`` – path to PNG file or ``None``

        Raises:
            :class:`AgentExecutionError` – when the seed paper cannot be
                resolved or the literature search tool is not initialised.
        """
        if not self.literature_search:
            raise AgentExecutionError(
                "Literature search tool not initialised; cannot build citation graph."
            )

        seed_identifier: str = context["seed_paper"]
        depth: int = int(context.get("depth", 2))
        max_refs: int = int(context.get("max_refs_per_paper", 30))
        include_citing: bool = bool(context.get("include_citing_papers", False))
        output_dir: str = context.get("output_dir", "tmp/citation_graph")
        do_save_dot: bool = bool(context.get("save_dot", False))
        do_save_png: bool = bool(context.get("save_png", False))

        os.makedirs(output_dir, exist_ok=True)
        pdf_dir = os.path.join(output_dir, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)

        graph = CitationGraph()

        # -----------------------------------------------------------
        # 1. Resolve seed paper → Semantic Scholar paper ID
        # -----------------------------------------------------------
        logger.info(f"[CitationGraph] Resolving seed paper: {seed_identifier!r}")
        seed_s2_id = await self.literature_search.resolve_paper_id(seed_identifier)
        if not seed_s2_id:
            raise AgentExecutionError(
                f"Could not resolve seed paper {seed_identifier!r} to a "
                f"Semantic Scholar paper ID."
            )
        logger.info(f"[CitationGraph] Seed S2 ID: {seed_s2_id}")

        # Build a minimal seed node from what we have so far; it will be
        # enriched during the deep-read pass.
        seed_node_dict: Dict[str, Any] = {
            "title": seed_identifier,
            "authors": [],
            "paper_id": seed_s2_id,
            "source": "semantic_scholar",
        }
        seed_key = graph.add_paper(seed_node_dict, depth=0)

        # -----------------------------------------------------------
        # 2. BFS expansion
        # -----------------------------------------------------------
        # frontier: list of (node_key, s2_paper_id, current_depth)
        frontier: List[Tuple[str, str, int]] = [(seed_key, seed_s2_id, 0)]
        visited_ids: Set[str] = {seed_s2_id}

        # Optionally seed the graph with citing papers (depth-0 inverse edges)
        if include_citing:
            logger.info("[CitationGraph] Fetching papers that cite the seed…")
            citing_papers = await self.literature_search.get_citing_papers(
                seed_s2_id, max_results=max_refs
            )
            for cp in citing_papers:
                cp_dict = {
                    "title": cp.title,
                    "authors": cp.authors,
                    "abstract": cp.abstract,
                    "year": cp.year,
                    "doi": cp.doi,
                    "journal": cp.journal,
                    "url": cp.url,
                    "citations": cp.citations,
                    "pdf_url": cp.pdf_url,
                    "source": cp.source,
                    "paper_id": cp.paper_id,
                }
                cp_key = graph.add_paper(cp_dict, depth=1)
                graph.add_citation(cp_key, seed_key)
                if cp.paper_id and cp.paper_id not in visited_ids and depth > 1:
                    visited_ids.add(cp.paper_id)
                    frontier.append((cp_key, cp.paper_id, 1))

        while frontier:
            node_key, s2_id, current_depth = frontier.pop(0)

            if current_depth >= depth:
                continue  # do not expand further than requested depth

            logger.info(
                f"[CitationGraph] Expanding depth {current_depth+1} "
                f"from node {node_key!r} (S2={s2_id})"
            )
            refs = await self.literature_search.get_references_by_paper_id(
                s2_id, max_results=max_refs
            )

            for ref in refs:
                ref_dict: Dict[str, Any] = {
                    "title": ref.title,
                    "authors": ref.authors,
                    "abstract": ref.abstract,
                    "year": ref.year,
                    "doi": ref.doi,
                    "journal": ref.journal,
                    "url": ref.url,
                    "citations": ref.citations,
                    "pdf_url": ref.pdf_url,
                    "source": ref.source,
                    "paper_id": ref.paper_id,
                }
                ref_key = graph.add_paper(ref_dict, depth=current_depth + 1)
                graph.add_citation(node_key, ref_key)

                if ref.paper_id and ref.paper_id not in visited_ids:
                    visited_ids.add(ref.paper_id)
                    frontier.append((ref_key, ref.paper_id, current_depth + 1))

        logger.info(
            f"[CitationGraph] BFS complete. "
            f"{len(graph)} nodes, {len(graph.node_keys())} unique papers."
        )

        # -----------------------------------------------------------
        # 3. Deep-read analysis for all papers with a URL / DOI
        # -----------------------------------------------------------
        deep_read_schema = {
            "type": "object",
            "properties": {
                "problem_and_background": {"type": "string"},
                "contributions": {"type": "string"},
                "methods": {"type": "string"},
                "challenges": {"type": "string"},
                "limitations_and_future_work": {"type": "string"},
            },
            "required": [
                "problem_and_background",
                "contributions",
                "methods",
                "challenges",
                "limitations_and_future_work",
            ],
        }

        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        async def _deep_read_node(nk: str) -> None:
            async with semaphore:
                node = graph.get_node(nk)
                if not node:
                    return

                # Skip if all analysis fields are already populated
                if all(node.get(ak) for ak in (
                    "problem_and_background", "contributions",
                    "methods", "challenges", "limitations_and_future_work"
                )):
                    return

                url = node.get("url")
                doi = node.get("doi")
                abstract = node.get("abstract") or ""

                # Prefer abstract-only analysis when no PDF is available; it
                # still yields useful structured output.
                text: Optional[str] = None

                if url or doi:
                    pdf_path = None
                    try:
                        if url:
                            pdf_path = download_pdf(url, save_folder=pdf_dir)
                        if doi and not pdf_path:
                            pdf_path = download_pdf_by_doi(
                                doi=doi, download_dir=pdf_dir
                            )
                        if pdf_path:
                            text = extract_text_from_pdf(pdf_path)
                    except Exception as e:
                        logger.debug(f"[CitationGraph] PDF fetch failed for {nk}: {e}")

                if not text and abstract:
                    text = abstract  # fall back to abstract

                if not text:
                    return

                prompt = (
                    f"Analyze the following academic paper content and extract "
                    f"structured information:\n\n{text[:_DEEP_READ_MAX_CHARS]}\n\n"
                    f"Return JSON with exactly these keys:\n"
                    f"- problem_and_background: the research problem and context\n"
                    f"- contributions: key contributions and solutions\n"
                    f"- methods: technical approach and methodology\n"
                    f"- challenges: challenges faced and open problems\n"
                    f"- limitations_and_future_work: limitations and future directions\n"
                    f"Return valid JSON only."
                )
                try:
                    analysis = await self._call_model(
                        prompt=prompt, schema=deep_read_schema
                    )
                    graph.update_analysis(nk, analysis)
                except Exception as e:
                    logger.warning(
                        f"[CitationGraph] LLM analysis failed for {nk}: {e}"
                    )

        # Run deep-read for all nodes concurrently (bounded by semaphore)
        tasks = [
            asyncio.create_task(_deep_read_node(nk))
            for nk in graph.node_keys()
        ]
        await asyncio.gather(*tasks)

        # -----------------------------------------------------------
        # 4. Serialise outputs
        # -----------------------------------------------------------
        json_path = os.path.join(output_dir, "citation_graph.json")
        graph.save_json(json_path)

        dot_path: Optional[str] = None
        png_path: Optional[str] = None

        if do_save_dot:
            dot_path = os.path.join(output_dir, "citation_graph.dot")
            graph.save_dot(dot_path)

        if do_save_png:
            png_path = os.path.join(output_dir, "citation_graph.png")
            if not graph.save_png(png_path):
                png_path = None

        # Flatten nodes to a list of dicts for callers that expect `papers`
        flat_papers = [graph.get_node(nk) for nk in graph.node_keys()]

        logger.info(
            f"[CitationGraph] Done. {len(flat_papers)} papers. "
            f"JSON → {json_path}"
        )

        return {
            "citation_graph": graph.to_json(),
            "papers": flat_papers,
            "graph_json_path": json_path,
            "graph_dot_path": dot_path,
            "graph_png_path": png_path,
        }
