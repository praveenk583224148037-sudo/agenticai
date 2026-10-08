"""Web Search AI Agent: Searches the web and provides comprehensive AI answers with sources."""

import argparse
import base64
import html
import json
import os
import re
import sys
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, urlsplit
from urllib.request import Request, urlopen


class BingResultsParser(HTMLParser):
    """Extract web results from a Bing search results page."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._result: dict[str, str] | None = None
        self._depth = 0
        self._in_heading = False
        self._in_title_link = False
        self._in_snippet = False
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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
        elif tag == "p" and not self._result["snippet"]:
            self._in_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if self._result is None:
            return

        if tag == "a" and self._in_title_link:
            self._in_title_link = False
            raw_title = "".join(self._title_parts).strip()
            self._result["title"] = html.unescape(" ".join(raw_title.split()))
        elif tag == "h2":
            self._in_heading = False
        elif tag in ("p", "div") and self._in_snippet:
            self._in_snippet = False
            raw_snippet = "".join(self._snippet_parts).strip()
            self._result["snippet"] = html.unescape(" ".join(raw_snippet.split()))

        self._depth -= 1
        if self._depth == 0:
            if self._result["title"] and self._result["url"]:
                self.results.append(self._result)
            self._result = None

    def handle_data(self, data: str) -> None:
        if self._in_title_link:
            self._title_parts.append(data)
        if self._in_snippet:
            self._snippet_parts.append(data)


def unwrap_bing_url(url: str) -> str:
    """Unwrap Bing's base64-encoded result redirect when present."""
    redirect_target = parse_qs(urlsplit(url).query).get("u", [""])[0]
    if not redirect_target.startswith("a1"):
        return url

    encoded_target = redirect_target[2:]
    encoded_target += "=" * (-len(encoded_target) % 4)
    try:
        decoded_target = base64.urlsafe_b64decode(encoded_target).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return url
    return decoded_target if decoded_target.startswith(("http://", "https://")) else url


_direct_result_url = unwrap_bing_url


def clean_query_fallback(query: str) -> str:
    """Strip conversational filler words for clean search engine queries."""
    cleaned = re.sub(
        r"^(tell me about|what is|what are|explain|who is|who are|how to|search for|information on|details about|can you tell me about)\s+",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip() or query


def call_gemini(prompt: str, models: tuple[str, ...] = ("gemini-3.8-flash", "gemini-3.5-flash")) -> str:
    """Call Google Gemini API using native urllib without external dependencies."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    last_error: Exception | None = None
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
        req = Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as err:
            last_error = err
            continue

    raise RuntimeError(f"Gemini API request failed: {last_error}")


def optimize_search_query(user_query: str) -> str:
    """Convert natural conversational user queries into optimal search engine keywords."""
    if not os.environ.get("GEMINI_API_KEY"):
        return clean_query_fallback(user_query)

    try:
        prompt = (
            "Convert the following user question into 1 to 4 optimal web search keywords for a search engine. "
            "Output ONLY the search query keywords, without quotes, punctuation, or explanations:\n"
            f"{user_query}"
        )
        optimized = call_gemini(prompt).strip(" \"'\n")
        return optimized if optimized else clean_query_fallback(user_query)
    except Exception:
        return clean_query_fallback(user_query)


def search_web(
    query: str,
    limit: int = 6,
    unwrap_urls: bool = True,
) -> list[dict[str, str]]:
    """Search Bing for relevant web pages with English market localization."""
    url = f"https://www.bing.com/search?q={quote_plus(query)}&setmkt=en-US&setlang=en-US"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": "SRCHHPGUSR=ADLT=OFF&NRSLT=10&SRCHLANG=en;",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            raw_page = response.read()
            charset = response.headers.get_content_charset()
            if charset:
                page = raw_page.decode(charset, errors="replace")
            else:
                try:
                    page = raw_page.decode("utf-8")
                except UnicodeDecodeError:
                    page = raw_page.decode("cp1252", errors="replace")
    except HTTPError as error:
        raise RuntimeError(f"Search provider returned HTTP {error.code}.") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach search provider: {error.reason}") from error
    except TimeoutError as error:
        raise RuntimeError("The web search request timed out.") from error

    parser = BingResultsParser()
    parser.feed(page)
    results = parser.results

    if unwrap_urls:
        for item in results:
            item["url"] = unwrap_bing_url(item["url"])

    if limit is not None and limit > 0:
        results = results[:limit]

    return results


# =====================================================================
# Tool Definitions for LLM Function Calling (Gemini & OpenAI Formats)
# =====================================================================

# 1. Gemini Function Declaration Tool Format
WEB_SEARCH_TOOL_JSON = {
    "function_declarations": [
        {
            "name": "web_search",
            "description": "Searches the live web to retrieve up-to-date information, news, links, and snippets on any topic.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": "The search keywords or query to look up on the web."
                    },
                    "limit": {
                        "type": "INTEGER",
                        "description": "The maximum number of search results to return (default is 6)."
                    }
                },
                "required": ["query"]
            }
        }
    ]
}

# 2. OpenAI / Compatible Tool Format
WEB_SEARCH_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Searches the live web to retrieve up-to-date information, news, links, and snippets on any topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search keywords or query to look up on the web."
                },
                "limit": {
                    "type": "integer",
                    "description": "The maximum number of search results to return (default is 6)."
                }
            },
            "required": ["query"]
        }
    }
}


def execute_web_search_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Execute the web_search tool when called by an LLM function call."""
    query = args.get("query", "")
    limit = int(args.get("limit", 6))
    results = search_web(query, limit=limit)
    return {
        "query": query,
        "count": len(results),
        "results": results
    }


def synthesize_ai_answer(user_query: str, results: list[dict[str, str]]) -> str:
    """Use Gemini AI to synthesize a comprehensive answer with citations from search results."""
    if not results:
        return "No web search results could be found for your query."

    context_blocks = []
    for idx, item in enumerate(results, start=1):
        context_blocks.append(
            f"Source [{idx}]:\n"
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Snippet: {item.get('snippet', '')}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = (
        "You are an expert Web Search AI Research Assistant.\n"
        "Your goal is to provide a comprehensive, accurate, and easy-to-read response to the user's question "
        "using the web search results provided below.\n\n"
        "Guidelines:\n"
        "- Explain the concepts clearly with appropriate headings, bullet points, and code/examples if relevant.\n"
        "- Cite sources accurately using bracketed numbers like [1], [2] throughout your answer.\n"
        "- Do not make up information that contradicts the search results.\n\n"
        f"User Question: {user_query}\n\n"
        f"Web Search Results:\n{context_text}\n\n"
        "Please provide the detailed answer:"
    )

    return call_gemini(prompt)


def run_web_ai(user_query: str, limit: int = 6) -> dict[str, Any]:
    """Execute full Web AI pipeline: optimize query -> search web -> synthesize answer."""
    search_keywords = optimize_search_query(user_query)
    results = search_web(search_keywords, limit=limit)

    ai_answer = ""
    error = None

    if os.environ.get("GEMINI_API_KEY"):
        try:
            ai_answer = synthesize_ai_answer(user_query, results)
        except Exception as e:
            error = str(e)
            ai_answer = f"(AI synthesis unavailable: {e})"
    else:
        ai_answer = "GEMINI_API_KEY is not set. Please set the environment variable to enable AI answers."

    return {
        "user_query": user_query,
        "search_keywords": search_keywords,
        "results": results,
        "ai_answer": ai_answer,
        "error": error,
    }


def format_cli_output(data: dict[str, Any], output_format: Literal["text", "json", "markdown"] = "text") -> str:
    """Format Web AI response for CLI presentation."""
    if output_format == "json":
        return json.dumps(data, indent=2, ensure_ascii=False)

    lines: list[str] = []

    if output_format == "markdown":
        lines.append(f"# Web AI: {data['user_query']}\n")
        lines.append(f"**Search Query Used:** `{data['search_keywords']}`\n")
        lines.append(data["ai_answer"])
        lines.append("\n### Sources\n")
        for idx, res in enumerate(data.get("results", []), start=1):
            lines.append(f"{idx}. [{res['title']}]({res['url']})")
            if res.get("snippet"):
                lines.append(f"   > {res['snippet']}")
        return "\n".join(lines)

    # Standard clean text presentation
    lines.append(f"\n=======================================================")
    lines.append(f"  WEB SEARCH AI: {data['user_query']}")
    lines.append(f"  Keywords: {data['search_keywords']}")
    lines.append(f"=======================================================\n")
    lines.append(data["ai_answer"])
    lines.append("\n-------------------------------------------------------")
    lines.append("Sources:")
    for idx, res in enumerate(data.get("results", []), start=1):
        lines.append(f"[{idx}] {res['title']}")
        lines.append(f"    URL: {res['url']}")
        if res.get("snippet"):
            lines.append(f"    {res['snippet']}")
        lines.append("")
    lines.append("-------------------------------------------------------")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Web Search AI: Searches the web and provides detailed AI answers with sources."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question or query for the Web AI to answer. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=6,
        help="Number of web sources to retrieve (default: 6).",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "-m",
        "--markdown",
        action="store_true",
        help="Output results as Markdown.",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Only display web search results without generating an AI answer.",
    )
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            sys.stdout.reconfigure(errors="replace")

    parser = build_parser()
    args = parser.parse_args()

    format_choice: Literal["text", "json", "markdown"] = "text"
    if args.json:
        format_choice = "json"
    elif args.markdown:
        format_choice = "markdown"

    initial_query = " ".join(args.query).strip()

    def process_query(query_str: str) -> None:
        if not query_str:
            return

        if args.search_only:
            keywords = optimize_search_query(query_str)
            results = search_web(keywords, limit=args.limit)
            if format_choice == "json":
                print(json.dumps({"query": query_str, "results": results}, indent=2, ensure_ascii=False))
            else:
                print(f"\nWeb Search Results for: {query_str} (Keywords: {keywords})\n")
                for i, r in enumerate(results, 1):
                    print(f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}\n")
            return

        if format_choice != "json":
            print(f"\n[Web AI] Searching the web for: \"{query_str}\"...")

        data = run_web_ai(query_str, limit=args.limit)
        print(format_cli_output(data, output_format=format_choice))

    if initial_query:
        process_query(initial_query)
        return 0

    # Interactive mode
    print("==================================================")
    print("      Welcome to Web Search AI Assistant          ")
    print("  Type your question or 'exit' / 'quit' to leave. ")
    print("==================================================")

    while True:
        try:
            user_input = input("\nAsk Web AI: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        process_query(user_input)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
