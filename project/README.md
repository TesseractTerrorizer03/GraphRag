# Graphrag — Quick README

All the milestone code for the Graphrag project is collected in the notebook
**`Graphrag_Implementation.ipynb`**.


---

## Project overview

This repo contains a RAG (Retrieval-Augmented Generation) pipeline that builds a
knowledge graph from text chunks and exposes a chat UI to query it.
The UI sends a text question to the backend, which performs semantic search on
the graph, builds a small explanatory subgraph, and then asks the LLM to
generate an answer grounded in those nodes.

---

## Important files / folders

* `Graphrag_Implementation.ipynb`
  Notebook with the full implementation, experiments, and milestone code.

* `ui_app/`
  Minimal UI + backend for querying the RAG:

  * `ui_app/app.py` — FastAPI app that serves the static chat UI and `/query` endpoint.
  * `ui_app/index.html` — Simple single-page chat UI (sends user question → `/query`).
  * `ui_app/rag_backend.py` — Async wrapper that calls into the in-project RAG functions and logs responses.
  * `ui_app/Query.py` — RAG glue used by the UI:

    * exposes the `rag` GraphRAG instance,
    * provides the `query_and_answer(rag, query, ...)` helper that runs semantic search → subgraph → LLM generation.
      (Concise role: `Query.py` provides the programmatic interface the UI uses to get a grounded answer.)

* `ui_app/query_logs/`
  Every question + full RAG response is saved sequentially as `question1.json`,
  `question2.json`, … The JSON includes the LLM answer plus the subgraph nodes
  and evidence used to craft that answer.

---

## How to run (local / development)

1. Open a terminal and `cd` to the project root (the folder that contains `ui_app`):

   ```bash
   cd project
   ```

2. Create & activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   .venv\Scripts\activate       # Windows (PowerShell)
   ```

3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI UI server:

   ```bash
   python ui_app/app.py
   ```

5. Open the UI in your browser. If your environment forwards port `8000`
   you may need to use the forwarded address shown by your Docker/IDE (e.g.
   `http://localhost:8088` if you forward `8000 → 8088`). The browser address should be the one listed in your VS Code **Ports** panel or Docker port forwarding.

---

## Notes & tips

* **Port forwarding**
  If you run inside Docker or Codespaces, make sure the host/port in `index.html`
  (the `fetch("http://localhost:8000/query")` call) matches the forwarded address
  your environment exposes to the browser. My local setup forwards `8000` to
  `http://localhost:8088`, so the browser fetch URL must point at `http://localhost:8088/query`.

* **Dependencies**
  The notebook and the UI share code. Install the same packages listed in
  `requirements.txt` used to run `Graphrag_Implementation.ipynb`.

* **Where the UI calls the RAG**
  `ui_app/rag_backend.py` imports `Query.py` from the project root to reuse the
  same `rag` instance and the `query_and_answer(...)` helper — this keeps the
  UI logic tiny and the heavy lifting inside the project's code.

* **Logs**
  All questions and the detailed RAG responses are saved to
  `ui_app/query_logs/question<N>.json`. These files contain:

  ```json
  {
    "question_number": N,
    "question": "...",
    "result": {
      "answer": "...",
      "used_concepts": [...],
      "references": [...],
      "subgraphs": [...]
    }
  }
  ```

---

## Example usage

1. Start the app:

   ```bash
   python ui_app/app.py
   ```

2. Open the UI in your browser (use the forwarded address if applicable),
   type a question and press **Send**.

3. The answer will appear in the chat UI. Each query is also written to
   `ui_app/query_logs` as a JSON file.
