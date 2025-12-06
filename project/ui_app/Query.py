import aiohttp
import asyncio
import json
import re
import textwrap
import ast
from pathlib import Path
from typing import List, Dict, Any, Union
import networkx as nx

OLLAMA_HOST = "http://host.docker.internal:11434"
EMBED_MODEL = "all-minilm:l6-v2"
CHAT_MODEL = "qwen2"

EMBED_URL = f"{OLLAMA_HOST}/api/embeddings"
CHAT_URL = f"{OLLAMA_HOST}/api/chat"

_session = None


async def get_session():
    global _session
    if _session is None:
        _session = aiohttp.ClientSession()
    return _session


async def embed_text_ollama(text_or_texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Accepts either a single string or a list of strings.
    Returns embedding vector or list of embedding vectors.
    """
    session = await get_session()

    async def embed_one(text: str):
        payload = {"model": EMBED_MODEL, "prompt": text}
        async with session.post(EMBED_URL, json=payload, timeout=60) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["embedding"]

    if isinstance(text_or_texts, str):
        return await embed_one(text_or_texts)

    tasks = [embed_one(t) for t in text_or_texts]
    return await asyncio.gather(*tasks)


embed_text_ollama.embedding_dim = 384


async def ollama_chat(prompt: str, system_prompt: str = None,
                      history_messages: List[Dict[str, str]] = [], **kwargs) -> str:
    """
    Sends a chat-style prompt to Ollama chat endpoint and returns the assistant text.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "num_predict": kwargs.get("num_predict", 2048),
    }

    session = await get_session()
    async with session.post(CHAT_URL, json=payload, timeout=2000) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["message"]["content"]


from nano_graphrag import GraphRAG

rag = GraphRAG(
    working_dir="./graphrag_website_new",
    cheap_model_func=ollama_chat,
    best_model_func=ollama_chat,
    embedding_func=embed_text_ollama,
)

MAX_CANDIDATES = 6
MAX_PREREQ_DEPTH = 4
MAX_SIBLINGS = 6
MAX_RESOURCES = 3
MAX_EXAMPLES = 3
MAX_PROMPT_CONCEPTS = 5


def _safe_trunc(s: str, n=300):
    if not s:
        return ""
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def _strip_quotes(s):
    return s.strip().strip('"').strip("'") if isinstance(s, str) else s


def _norm(s: str):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _ensure_nx_graph(rag):
    storage = getattr(rag, "chunk_entity_relation_graph", None)
    if storage is not None:
        # common wrappers
        for cand in ("nx_graph", "G", "graph", "networkx_graph"):
            g = getattr(storage, cand, None)
            if g is not None:
                return nx.DiGraph(g)
        for method in ("get_graph", "to_networkx"):
            if hasattr(storage, method):
                try:
                    g = getattr(storage, method)()
                    return nx.DiGraph(g)
                except Exception:
                    pass

    wk = getattr(rag, "working_dir", "./graphrag_website_new")
    for p in (Path(wk) / "graph_chunk_entity_relation.graphml",
              Path(wk) / "graph_chunk_entity_relation.graphml.gz",
              Path("./graphrag_website/graph_chunk_entity_relation.graphml")):
        if p.exists():
            try:
                g = nx.read_graphml(str(p))
                return nx.DiGraph(g)
            except Exception:
                pass
    raise RuntimeError("Could not load networkx graph from rag.working_dir or wrapper")


_KV_CHUNKS_CACHE: Dict[str, Dict[str, Any]] = None


def load_kv_chunks_map(rag) -> Dict[str, Dict[str, Any]]:
    """
    Return mapping chunk_id -> chunk record (dict). The file typically is kv_store_text_chunks.json
    """
    global _KV_CHUNKS_CACHE
    if _KV_CHUNKS_CACHE is not None:
        return _KV_CHUNKS_CACHE
    wk = getattr(rag, "working_dir", None) or "./graphrag_website_new"
    cand_paths = [
        Path(wk) / "kv_store_text_chunks.json",
        Path("./graphrag_website_new/kv_store_text_chunks.json"),
        Path("./kv_store_text_chunks.json"),
    ]
    for p in cand_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    _KV_CHUNKS_CACHE = {d.get("chunk_id") or d.get("id") or d.get("_id"): d for d in data if isinstance(d, dict)}
                elif isinstance(data, dict):
                    _KV_CHUNKS_CACHE = data
                else:
                    _KV_CHUNKS_CACHE = {}
                return _KV_CHUNKS_CACHE
            except Exception as e:
                print("load_kv_chunks_map: failed to parse", p, e)
    _KV_CHUNKS_CACHE = {}
    return _KV_CHUNKS_CACHE


def chunk_snippet_from_source_id(source_id: str, kv_chunks: Dict[str, Dict[str, Any]], chars=400) -> str:
    if not source_id:
        return ""
    parts = re.split(r'\<SEP\>|\||,|\s+', str(source_id))
    snippets = []
    for p in parts:
        if not p:
            continue
        rec = kv_chunks.get(p)
        if not rec:
            continue
        text = rec.get("chunk_text") or rec.get("text") or rec.get("content") or rec.get("body") or ""
        if text:
            snippets.append(_safe_trunc(text.strip(), chars))
    return "\n\n".join(snippets[:2]) if snippets else ""


async def semantic_candidates(rag, query: str, top_k: int = MAX_CANDIDATES) -> List[Dict[str, Any]]:
    qs = query.strip()

    try:
        if hasattr(rag, "entities_vdb") and hasattr(rag.entities_vdb, "query"):
            res = await rag.entities_vdb.query(qs, top_k=top_k)
            if res is None:
                raise RuntimeError("entities_vdb.query returned None")
            candidates = []
            for r in res:
                node_id = r.get("id") or r.get("_id") or r.get("entity_id") or r.get("entity_name") or r.get("meta", {}).get("entity_name")
                score = r.get("score") if "score" in r else r.get("distance") or r.get("sim") or None
                candidates.append({"node_id": node_id, "score": score, "raw": r})
            candidates = [c for c in candidates if c["node_id"]]
            try:
                candidates = sorted(candidates, key=lambda x: float(x.get("score")) if x.get("score") is not None else 0.0, reverse=True)
            except Exception:
                pass
            if candidates:
                print(f"[debug] semantic_candidates (via entities_vdb.query) -> {[c['node_id'] for c in candidates]}")
                return candidates[:top_k]
    except Exception as e:
        print("[debug] entities_vdb.query failed:", e)

    try:
        emb_func = getattr(rag, "embedding_func", None) or getattr(rag, "embedding", None)
        if emb_func is None:
            emb_func = getattr(rag.entities_vdb, "embedding_func", None)
        if emb_func is None:
            raise RuntimeError("No embedding function found on rag")

        if asyncio.iscoroutinefunction(emb_func):
            emb = await emb_func(qs)
        else:
            loop = asyncio.get_event_loop()
            emb = await loop.run_in_executor(None, emb_func, qs)

        if isinstance(emb, list) and len(emb) and isinstance(emb[0], list):
            emb_vec = emb[0]
        else:
            emb_vec = emb

        client = getattr(rag.entities_vdb, "_client", None)
        if client is None:
            raise RuntimeError("No low-level _client on entities_vdb")

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, client.query, emb_vec, top_k, getattr(rag.entities_vdb, "cosine_better_than_threshold", None))
        cand = []
        for r in results:
            nid = r.get("id") or r.get("_id") or r.get("meta", {}).get("entity_name") or r.get("entity_name")
            score = r.get("score") or r.get("distance") or None
            cand.append({"node_id": nid, "score": score, "raw": r})
        if cand:
            print(f"[debug] semantic_candidates (via low-level client) -> {[c['node_id'] for c in cand]}")
            return cand[:top_k]
    except Exception as e:
        print("[debug] fallback embedding->client failed:", e)

    try:
        G = _ensure_nx_graph(rag)
        node_meta = {n: dict(G.nodes[n]) for n in G.nodes()}
        sample = []
        for nid, md in list(node_meta.items())[:200]:
            title = md.get("title") or md.get("entity_name") or nid
            defs = md.get("definitions") or md.get("definition") or md.get("description") or ""
            sample.append(f"{title} ||| id={nid} ||| def={_safe_trunc(defs,120)}")
        nodes_context = "\n".join(sample)
        prompt = textwrap.dedent(f"""
            Choose up to {top_k} node ids from the following list that best match the student's question.
            Return a JSON array of ids only.

            KG_NODES:
            {nodes_context}

            Student question:
            {query}
        """)
        best_fn = getattr(rag, "best_model_func", None) or getattr(rag, "cheap_model_func", None)
        if best_fn is None:
            raise RuntimeError("No LLM function is available on rag")
        hk = getattr(rag, "llm_response_cache", None)
        resp = await best_fn(prompt, system_prompt=None, history_messages=[], hashing_kv=hk)
        cand_ids = []
        try:
            cand_ids = json.loads(resp)
            if not isinstance(cand_ids, list):
                cand_ids = []
        except Exception:
            cand_ids = re.findall(r'["\']([A-Za-z0-9_\-\. ]+)["\']', resp)
        out = [{"node_id": cid, "score": None, "raw": None} for cid in cand_ids if cid]
        print("[debug] LLM fallback candidates:", [c['node_id'] for c in out])
        return out[:top_k]
    except Exception as e:
        print("[debug] final fallback failed:", e)
        return []


def resolve_candidate_to_node_id(rag, candidate: Dict[str, Any]):
    """
    Map a vector-candidate to a graph node id using many heuristics.
    """
    try:
        G = _ensure_nx_graph(rag)
    except Exception as e:
        print("[resolve] cannot load graph:", e)
        return None
    nodes = list(G.nodes())

    title_map = {}
    for n in nodes:
        md = dict(G.nodes[n])
        for key in ("title", "entity_name", "name", "label", "text"):
            val = md.get(key)
            if isinstance(val, str) and val.strip():
                title_map[_norm(_strip_quotes(val))] = n
        title_map[_norm(_strip_quotes(str(n)))] = n
        for k in ("vector_id", "vdb_id", "ent_id", "entity_id", "_id", "id"):
            if k in md and md[k] is not None:
                title_map[_norm(str(md[k]))] = n

    raw = candidate.get("raw", {}) or {}
    candidate_node_id = candidate.get("node_id")

    if candidate_node_id in nodes:
        return candidate_node_id

    for key in ("entity_node_id", "node_id", "node", "graph_node_id", "map_to"):
        if key in raw and raw[key] in nodes:
            return raw[key]

    meta = raw.get("meta") or raw.get("metadata") or raw
    if isinstance(meta, dict):
        for key in ("entity_name", "title", "name", "label", "id"):
            val = meta.get(key)
            if isinstance(val, str) and _norm(_strip_quotes(val)) in title_map:
                return title_map[_norm(_strip_quotes(val))]

    if isinstance(candidate_node_id, str):
        cand_norm = _norm(_strip_quotes(candidate_node_id))
        if cand_norm in title_map:
            return title_map[cand_norm]
        for k, v in title_map.items():
            if cand_norm and (cand_norm in k or k in cand_norm):
                return v

    for fk in ("source", "file", "source_url", "chunk_file", "_file"):
        val = meta.get(fk) if isinstance(meta, dict) else None
        if val:
            val_norm = _norm(str(val))
            for n in nodes:
                md = dict(G.nodes[n])
                for nk in ("source", "file", "source_url", "chunk_file", "_file"):
                    if _norm(str(md.get(nk, ""))) == val_norm:
                        return n

    search_text = ""
    for k in ("text", "chunk_text", "content", "title", "description"):
        if k in raw and isinstance(raw[k], str):
            search_text += " " + raw[k]
    search_text = _norm(search_text)
    if search_text:
        for k, v in title_map.items():
            if k and k in search_text:
                return v

    return None


def build_subgraph_for_node(rag, node_id: str) -> Dict[str, Any]:
    G = _ensure_nx_graph(rag)
    if node_id not in G:
        print(f"[debug] node_id {node_id} not found in graph")
        return {}

    nodes = {}
    edges = []

    nodes[node_id] = dict(G.nodes[node_id])
    central_meta = nodes[node_id]

    prereqs = []
    seen = set()

    def walk_prereq(n, depth):
        if depth <= 0:
            return
        for u, v, d in G.in_edges(n, data=True):
            et = (d.get("type") or d.get("relation") or d.get("label") or "").lower()
            if "prereq" in et or "requires" in et or "precedes" in et:
                if u not in seen:
                    seen.add(u)
                    nodes[u] = dict(G.nodes[u])
                    edges.append({"source": u, "target": n, "type": et or "prereq"})
                    prereqs.append(u)
                    walk_prereq(u, depth - 1)

    walk_prereq(node_id, MAX_PREREQ_DEPTH)
    prereqs = list(reversed(prereqs))

    siblings = []
    for u, v, d in list(G.edges(node_id, data=True)):
        et = (d.get("type") or d.get("relation") or d.get("label") or "").lower()
        if any(k in et for k in ("near", "similar", "related", "sibling", "contrast", "part_of", "is_a")):
            if v not in nodes:
                nodes[v] = dict(G.nodes[v])
            edges.append({"source": node_id, "target": v, "type": et or "related"})
            siblings.append(v)
    if len(siblings) < 1:
        for nbr in list(G.neighbors(node_id))[:MAX_SIBLINGS]:
            if nbr not in nodes:
                nodes[nbr] = dict(G.nodes[nbr])
            edges.append({"source": node_id, "target": nbr, "type": "neighbor"})
            siblings.append(nbr)
    siblings = siblings[:MAX_SIBLINGS]

    resources = []
    for u, v, d in G.in_edges(node_id, data=True):
        et = (d.get("type") or d.get("relation") or d.get("label") or "").lower()
        if any(k in et for k in ("explain", "reference", "source", "cite", "mentions")) or str(G.nodes[u].get("type", "")).lower().startswith("resource"):
            nodes[u] = dict(G.nodes[u])
            edges.append({"source": u, "target": node_id, "type": et or "explains"})
            resources.append(u)
    resources = resources[:MAX_RESOURCES]

    examples = []
    for u, v, d in G.in_edges(node_id, data=True):
        et = (d.get("type") or d.get("relation") or d.get("label") or "").lower()
        if "example" in et or str(G.nodes[u].get("type", "")).lower() == "example":
            nodes[u] = dict(G.nodes[u])
            edges.append({"source": u, "target": node_id, "type": et or "exemplifies"})
            examples.append(u)
    examples = examples[:MAX_EXAMPLES]

    sub = {
        "concept_id": node_id,
        "concept_meta": central_meta,
        "prereqs": prereqs,
        "siblings": siblings,
        "resources": resources,
        "examples": examples,
        "nodes": nodes,
        "edges": edges,
    }

    kv = load_kv_chunks_map(rag)
    if not sub["resources"]:
        possible_source = central_meta.get("source_id") or central_meta.get("source") or central_meta.get("_file") or central_meta.get("file")
        snippet = chunk_snippet_from_source_id(possible_source or "", kv, chars=400)
        if snippet:
            fake_id = f"chunk::{(possible_source or 'unknown')}"
            sub["nodes"][fake_id] = {
                "title": f"source snippet for {strip_outer_quotes(node_id)}",
                "text": snippet,
                "source": possible_source
            }
            sub["resources"].append(fake_id)
            sub["edges"].append({"source": fake_id, "target": node_id, "type": "explains (chunk-snippet)"})

    return sub


def strip_outer_quotes(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1].strip()
    return s


def parse_llm_json_response(resp_text: str, max_unwrap=2):
    if not isinstance(resp_text, str):
        return None
    attempt = resp_text
    for _ in range(max_unwrap + 1):
        try:
            parsed = json.loads(attempt)
        except Exception:
            try:
                parsed = ast.literal_eval(attempt)
            except Exception:
                parsed = None
        if isinstance(parsed, dict):
            for k, v in list(parsed.items()):
                if isinstance(v, str) and v.strip().startswith(('{', '[')) and v.strip().endswith(('}', ']')):
                    try:
                        parsed[k] = json.loads(v)
                    except Exception:
                        pass
            return parsed
        if isinstance(parsed, str):
            attempt = parsed
            continue
        break
    return None


async def generate_answer(rag, query: str, candidate_nodes: List[str], learner_level="introductory"):
    G = _ensure_nx_graph(rag)
    kv_map = load_kv_chunks_map(rag)

    subgraphs = []
    references = []
    for nid in candidate_nodes[:MAX_PROMPT_CONCEPTS]:
        sg = build_subgraph_for_node(rag, nid)
        if not sg:
            continue
        subgraphs.append(sg)

        meta = sg.get("concept_meta", {})
        title = strip_outer_quotes(meta.get("title") or meta.get("entity_name") or nid)
        short_def = _safe_trunc(strip_outer_quotes(meta.get("definitions") or meta.get("description") or meta.get("text") or ""), 300)

        evidence_text = ""
        if sg.get("resources"):
            first_r = sg["resources"][0]
            rmeta = sg["nodes"].get(first_r, {})
            evidence_text = rmeta.get("text") or rmeta.get("chunk_text") or ""
            if not evidence_text:
                src = rmeta.get("source") or rmeta.get("source_id") or rmeta.get("file") or meta.get("source_id")
                if src:
                    evidence_text = chunk_snippet_from_source_id(src, kv_map, chars=350)
            if not evidence_text:
                evidence_text = strip_outer_quotes(rmeta.get("description") or rmeta.get("definitions") or "")
        references.append({"concept": title, "evidence": _safe_trunc(evidence_text, 350) if evidence_text else ""})

    concept_sections = []
    for sg in subgraphs:
        nid = sg["concept_id"]
        meta = sg["concept_meta"]
        title = strip_outer_quotes(meta.get("title") or meta.get("entity_name") or nid)
        short_def = _safe_trunc(strip_outer_quotes(meta.get("definitions") or meta.get("description") or meta.get("text") or ""), 250)
        evids = []
        for rnode in sg.get("resources", [])[:2]:
            rmeta = sg["nodes"].get(rnode, {})
            ev = rmeta.get("text") or rmeta.get("chunk_text") or rmeta.get("description") or ""
            if not ev:
                src = rmeta.get("source") or rmeta.get("source_id") or meta.get("source_id")
                if src:
                    ev = chunk_snippet_from_source_id(src, kv_map, chars=300)
            if ev:
                evids.append(_safe_trunc(ev, 300))
        concept_sections.append(f"### {title}\n{short_def}\n\nEvidence:\n" + ("\n\n".join(evids) if evids else "(no snippet available)") + "\n\n")

    prompt = f"""
You are an expert AI tutor. Your job is to answer any technical questions the student asks using 
clear reasoning, with mathematics and code examples whenever they meaningfully improve understanding.

Where appropriate:
- Explain concepts first in simple terms, then add technical depth.
- Provide mathematical expressions for key equations but they must not be in latex
- Provide short Python examples when they clarify the mechanism or computation.
- If concepts involve comparisons, mechanisms, steps, or inner workings, give structured paragraphs.

You must base your answers ONLY on the provided concept summaries and exact evidence snippets 
extracted from the knowledge graph. Do NOT invent new facts.

OUTPUT FORMAT:
1. Answer to the question which includes an explanation, mathematical equation (optional) and a code snippet (optional)
2. List the knowledge graph nodes used.
3. Provide 1-2 short practice questions.


Student asked:
\"\"\"{query}\"\"\"

Learner level: {learner_level}

Concepts available:
{''.join(concept_sections)}
"""

    best_fn = getattr(rag, "best_model_func", None) or getattr(rag, "cheap_model_func", None)
    if best_fn is None:
        raise RuntimeError("No LLM function available on rag for generation")
    resp = await best_fn(prompt, system_prompt=None, history_messages=[], hashing_kv=getattr(rag, "llm_response_cache", None))

    parsed = parse_llm_json_response(resp)
    text_answer = None
    if isinstance(parsed, dict) and parsed.get("answer"):
        text_answer = parsed["answer"]
    elif isinstance(parsed, dict) and any(isinstance(v, str) for v in parsed.values()):
        joinable = " ".join([v for v in parsed.values() if isinstance(v, str)])
        text_answer = joinable.strip() if joinable.strip() else None
    if not text_answer:
        text_answer = resp.strip()

    return {
        "answer": text_answer,
        "used_concepts": [sg["concept_id"] for sg in subgraphs],
        "references": references,
        "subgraphs": subgraphs
    }


async def query_and_answer(rag, query: str, top_k=MAX_CANDIDATES, learner_level="introductory"):
    """
    Full pipeline: semantic retrieval -> resolve -> subgraph -> prompt -> LLM
    Returns structured dict {answer, used_concepts, references, subgraphs}
    """
    cand = await semantic_candidates(rag, query, top_k=top_k)
    if not cand:
        return {"answer": "No semantic candidates found", "used_concepts": [], "references": [], "subgraphs": []}

    resolved = []
    for c in cand:
        try:
            node = resolve_candidate_to_node_id(rag, c)
        except Exception as e:
            print("[warn] resolver error:", c.get("node_id"), e)
            node = None
        if node and node not in resolved:
            resolved.append(node)

    if not resolved:
        candidate_ids = [c["node_id"] for c in cand if c.get("node_id")]
        return await generate_answer(rag, query, candidate_ids, learner_level=learner_level)

    return await generate_answer(rag, query, resolved, learner_level=learner_level)


__all__ = ["rag", "query_and_answer", "generate_answer", "semantic_candidates", "resolve_candidate_to_node_id"]
