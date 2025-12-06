import sys
import os
import asyncio
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from project.ui_app.Query import rag, query_and_answer

LOG_DIR = os.path.join(PROJECT_ROOT, "ui_app", "query_logs")
COUNTER_FILE = os.path.join(LOG_DIR, "counter.txt")

os.makedirs(LOG_DIR, exist_ok=True)

def get_next_log_filename(): #global counter for questions
    if not os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "w") as f:
            f.write("1")
        return os.path.join(LOG_DIR, "question1.json"), 1

    with open(COUNTER_FILE, "r") as f:
        num = int(f.read().strip())

    next_num = num + 1

    with open(COUNTER_FILE, "w") as f:
        f.write(str(next_num))

    return os.path.join(LOG_DIR, f"question{next_num}.json"), next_num


async def query_rag(question: str):
    try:
        result = await query_and_answer(
            rag, question, top_k=6, learner_level="intermediate"
        )

        # print("\n====== FULL RAG RESULT ======")
        # print(json.dumps(result, indent=2))
        # print("====== END ======\n")

        filename, num = get_next_log_filename()

        log_data = {
            "question_number": num,
            "question": question,
            "result": result
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        print(f"[Saved] {filename}")

        return result.get("answer", "No answer generated.")

    except Exception as e:
        print("[error in query_rag]", e)
        return f"Error: {e}"
