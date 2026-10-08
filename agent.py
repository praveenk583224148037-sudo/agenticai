import io
import os
import sys
import json
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Ensure UTF-8 output encoding on Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# 1. Web Search Function
try:
    from webagent2 import search_web
except ImportError:
    import html
    from html.parser import HTMLParser
    from urllib.parse import quote_plus
    from urllib.request import Request, urlopen

    class BingResultsParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self._result = None
            self._depth = 0
            self._in_heading = False
            self._in_title_link = False
            self._in_snippet = False
            self._title_parts = []
            self._snippet_parts = []

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            classes = (attributes.get("class") or "").split()
            if tag == "li" and "b_algo" in classes:
                self._result = {"title": "", "url": "", "snippet": ""}
                self._depth = 1
                self._title_parts = []
                self._snippet_parts = []
                return
            if self._result is None:
                return
            self._depth += 1
            if tag == "h2":
                self._in_heading = True
            elif tag == "a" and self._in_heading:
                self._in_title_link = True
                self._result["url"] = attributes.get("href") or ""
            elif tag in ("p", "div") and "b_caption" in classes:
                self._in_snippet = True

        def handle_endtag(self, tag):
            if self._result is None:
                return
            if tag == "a" and self._in_title_link:
                self._in_title_link = False
                self._result["title"] = html.unescape(" ".join("".join(self._title_parts).split()))
            elif tag == "h2":
                self._in_heading = False
            elif tag in ("p", "div") and self._in_snippet:
                self._in_snippet = False
                self._result["snippet"] = html.unescape(" ".join("".join(self._snippet_parts).split()))
            self._depth -= 1
            if self._depth == 0:
                if self._result["title"] and self._result["url"]:
                    self.results.append(self._result)
                self._result = None

        def handle_data(self, data):
            if self._in_title_link:
                self._title_parts.append(data)
            if self._in_snippet:
                self._snippet_parts.append(data)

    def search_web(query, limit=5):
        url = f"https://www.bing.com/search?q={quote_plus(query)}&setmkt=en-US&setlang=en-US"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"})
        try:
            with urlopen(req, timeout=15) as resp:
                parser = BingResultsParser()
                parser.feed(resp.read().decode("utf-8", errors="replace"))
                return parser.results[:limit]
        except Exception as e:
            return [{"title": "Error", "url": "", "snippet": str(e)}]

# 2. Document Search & Student Database Indexing
def _load_document_records():
    """Loads student records from student_data.txt or student_database.pdf."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(base_dir, "student_data.txt")
    pdf_path = os.path.join(base_dir, "student_database.pdf")

    records = []
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str.startswith("ID: STU"):
                    records.append(line_str)
    elif os.path.exists(pdf_path):
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf_path)
            full_text = "\n".join([page.extract_text() for page in reader.pages])
            lines = [l.strip() for l in full_text.splitlines() if l.strip()]
            i = 0
            while i < len(lines):
                if lines[i].startswith("STU"):
                    try:
                        rec = (f"ID: {lines[i]} | Name: {lines[i+1]} | Dept: {lines[i+2]} | Year: {lines[i+3]} | "
                               f"Email: {lines[i+4]} | Phone: {lines[i+5]} | Attendance: {lines[i+6]} | "
                               f"Python: {lines[i+7]} | DSA: {lines[i+8]} | AI: {lines[i+9]} | Overall: {lines[i+10]} | "
                               f"Placement: {lines[i+11]}")
                        records.append(rec)
                        i += 12
                    except IndexError:
                        break
                else:
                    i += 1
        except Exception:
            pass
    return records

STUDENT_RECORDS = _load_document_records()

# Setup BM25 retriever for document search
_BM25_RETRIEVER = None
if STUDENT_RECORDS:
    try:
        from langchain_community.retrievers import BM25Retriever
        doc_objects = [Document(page_content=r) for r in STUDENT_RECORDS]
        _BM25_RETRIEVER = BM25Retriever.from_documents(doc_objects, k=6)
    except Exception:
        _BM25_RETRIEVER = None

# 3. LangChain Tool Definitions
@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Performs basic arithmetic calculations: 'add', 'subtract', 'multiply', or 'divide'."""
    try:
        a = float(a)
        b = float(b)
        if operation == "add":
            return str(a + b)
        elif operation == "subtract":
            return str(a - b)
        elif operation == "multiply":
            return str(a * b)
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero."
            return str(a / b)
        else:
            return f"Error: Unsupported operation '{operation}'."
    except Exception as e:
        return f"Error: {e}"

@tool
def web_search_tool(query: str) -> str:
    """Searches the live web to retrieve current information, facts, news, and details on any topic."""
    results = search_web(query, limit=5)
    return json.dumps(results, ensure_ascii=False)

@tool
def get_current_datetime() -> str:
    """Gets the current live date, time, and day of the week."""
    now = datetime.now()
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A"),
        "formatted": now.strftime("%A, %B %d, %Y %I:%M:%S %p")
    })

@tool
def document_search_tool(query: str) -> str:
    """Searches the student database document containing 25 student records, including student ID, name, department, email, phone, attendance %, marks (Python, DSA, AI, Overall), and placement eligibility.
    Use this tool whenever the user asks about students, marks, academic scores, departments, attendance, placement eligibility, or student contact details."""
    if not STUDENT_RECORDS:
        return "No document records are currently loaded."

    q_lower = query.lower()

    # For queries requiring full list, statistics, highest/lowest marks, department lists, or counts
    broad_keywords = ["highest", "lowest", "all", "every", "list", "how many", "count", "top", "rank", "database", "students", "average", "eligible", "below", "above"]
    if any(k in q_lower for k in broad_keywords):
        return "Complete Student Database Records:\n" + "\n".join(STUDENT_RECORDS)

    # Use BM25 retriever for specific student / topic searches
    if _BM25_RETRIEVER is not None:
        try:
            results = _BM25_RETRIEVER.invoke(query)
            if results:
                return "\n".join([doc.page_content for doc in results])
        except Exception:
            pass

    # Fallback keyword match
    tokens = [t for t in q_lower.split() if len(t) > 2]
    matched = [r for r in STUDENT_RECORDS if any(t in r.lower() for t in tokens)]
    if matched:
        return "\n".join(matched[:8])

    return "No matching student records found in the document."

TOOLS = [calculator, web_search_tool, get_current_datetime, document_search_tool]

ACTIVE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash"
]

def extract_text(content):
    """Helper to extract clean text from LangChain message content."""
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(item.get("text", ""))
            elif isinstance(item, str):
                text_chunks.append(item)
        return "\n".join(text_chunks).strip()
    return str(content)

def run_react_agent(api_key, combined_query):
    system_prompt = (
        "You are an expert AI agent built on the ReAct (Reasoning + Acting) framework using LangChain. "
        "You solve user questions step-by-step through multi-step reasoning. "
        "You can execute more than two sequential or dependent steps if needed to arrive at the final answer. "
        "Always use the available tools (calculator, web_search_tool, get_current_datetime, document_search_tool) when necessary. "
        "When answering questions about the student database or documents, use document_search_tool to retrieve the data first."
    )

    for model_name in ACTIVE_MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0
            )

            agent = create_react_agent(llm, TOOLS, prompt=system_prompt)

            print("\n" + "="*60)
            print("[LangChain ReAct Agent - Multi-Step Execution]")
            print("="*60)

            step_count = 0
            final_answer = ""

            # Stream step updates as each node executes
            for chunk in agent.stream({"messages": [HumanMessage(content=combined_query)]}, stream_mode="updates"):
                for node_name, update in chunk.items():
                    messages = update.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    step_count += 1
                                    print(f"\n[Step {step_count}]")
                                    print(f"[Action]: {tc['name']}")
                                    print(f"[Action Input]: {json.dumps(tc['args'], ensure_ascii=False)}")
                            elif msg.content:
                                final_answer = extract_text(msg.content)
                        elif isinstance(msg, ToolMessage):
                            obs_text = str(msg.content)
                            if len(obs_text) > 300:
                                obs_text = obs_text[:300] + "... [truncated]"
                            print(f"[Observation]: {obs_text}")

            if final_answer:
                print(f"\n[Thought]: All steps completed. Formulating final answer.")
                print(f"\n[Agent Final Answer]:\n{final_answer}")
                print("="*60)
                return
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "503" in err_str:
                continue
            else:
                print(f"Agent Execution Error: {e}")
                return

    print("Error: All fallback models are currently unavailable.")

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API Key: ").strip()

    print("\n" + "="*50)
    print("      LangChain Multi-Tool ReAct Agent")
    print("="*50)
    print("Available Tools:")
    print(" 1. Math / Calculator")
    print(" 2. Live Web Search")
    print(" 3. Live Date & Time")
    print(" 4. Document Search (Student Database)")
    print("(Press Enter to skip any input you don't need)\n")

    # 4 Separate inputs for Math, Web Search, Date/Time, and Document Search
    math_query = input("Enter your question (math): ").strip()
    web_query = input("Enter your question (web search): ").strip()
    datetime_query = input("Enter your question (date & time): ").strip()
    doc_query = input("Enter your question (document search): ").strip()

    query_parts = []
    if math_query:
        query_parts.append(f"Math question: {math_query}")
    if web_query:
        query_parts.append(f"Web search question: {web_query}")
    if datetime_query:
        query_parts.append(f"Date & time question: {datetime_query}")
    if doc_query:
        query_parts.append(f"Document search question: {doc_query}")

    if not query_parts:
        print("No query provided. Please enter at least one question.")
        return

    combined_query = "\n".join(query_parts)
    run_react_agent(api_key, combined_query)

if __name__ == "__main__":
    main()
