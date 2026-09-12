"""HybridSight agent: routes questions to RAG, web, Wikipedia, and vision."""

import re
import urllib.parse
import urllib.request
import json

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from config import get_llm
from tools_rag import search_documents
from tools_vision import describe_image


llm = get_llm()
duckduckgo = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the live web for current or recent information."""
    try:
        result = duckduckgo.invoke(query)
        if not result:
            return "No useful web results found."
        return str(result)
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for general encyclopedic knowledge."""
    try:
        encoded = urllib.parse.quote(query)
        search_url = (
            "https://en.wikipedia.org/w/api.php"
            "?action=query&list=search&format=json&utf8=1"
            "&srlimit=3&srsearch=" + encoded
        )
        request = urllib.request.Request(
            search_url,
            headers={"User-Agent": "HybridSight/1.0"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        hits = data.get("query", {}).get("search", [])
        if not hits:
            return f"No useful Wikipedia results found for: {query}"

        title = hits[0].get("title", query)
        title_encoded = urllib.parse.quote(title)
        article_url = (
            "https://en.wikipedia.org/w/api.php"
            "?action=query&prop=extracts&exintro=1"
            "&explaintext=1&format=json&redirects=1"
            "&titles=" + title_encoded
        )
        request = urllib.request.Request(
            article_url,
            headers={"User-Agent": "HybridSight/1.0"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            article_data = json.loads(response.read().decode("utf-8"))

        pages = article_data.get("query", {}).get("pages", {})
        if not pages:
            return f"Wikipedia found '{title}', but could not retrieve its content."

        page = next(iter(pages.values()))
        extract = page.get("extract", "").strip()
        if not extract:
            return f"Wikipedia found '{title}', but no article summary was available."

        return f"Wikipedia article: {title}\n\n{extract}"

    except Exception as e:
        return f"Wikipedia search failed: {e}"


def is_document_question(question: str) -> bool:
    q = question.lower().strip()
    patterns = [
        "my pdf", "my document", "my file",
        "the pdf", "the document", "the uploaded pdf",
        "the uploaded document", "the uploaded file",
        "uploaded pdf", "uploaded document", "uploaded file",
        "according to my pdf", "according to the pdf",
        "according to my document", "according to the document",
        "in my pdf", "in the pdf", "in my document", "in the document",
        "from my pdf", "from the pdf", "from my document", "from the document",
        "pdf content", "pdf contents", "document content", "document contents",
        "what does my pdf say", "what does the pdf say",
        "what does my document say", "what does the document say",
        "summarize my pdf", "summarise my pdf", "summarize the pdf", "summarise the pdf",
        "summarize my document", "summarise my document",
        "summarize the document", "summarise the document",
    ]
    return any(p in q for p in patterns)


def is_image_question(question: str) -> bool:
    q = question.lower().strip()
    patterns = [
        "this image", "the image", "uploaded image", "in this image", "in the image",
        "what is shown", "what can you see", "what do you see",
        "describe this image", "describe the image",
        "analyze this image", "analyse this image",
        "analyze the image", "analyse the image",
        "objects in the image", "objects are visible",
        "read this image", "read the image",
    ]
    return any(p in q for p in patterns)


def is_current_question(question: str) -> bool:
    q = question.lower().strip()
    patterns = [
        "latest", "current", "currently", "recent", "today", "tonight",
        "this week", "this month", "breaking news", "current news", "recent news",
        "latest news", "recent developments", "latest developments",
        "what happened", "right now", "as of now",
    ]
    return any(p in q for p in patterns)


def llm_route(question: str) -> str:
    prompt = f"""
Classify this HybridSight question into exactly one route.
Return ONLY one word: PDF, VISION, WEB, WIKIPEDIA, or MULTI.

PDF = explicitly about the user's uploaded PDF/document/file.
VISION = explicitly about an uploaded image.
WEB = current/latest/recent/today/breaking/changing information.
WIKIPEDIA = general knowledge, people, biographies, history, science,
technology, places, concepts, and encyclopedic facts.
MULTI = genuinely requires both uploaded PDF and image.

Important: an available PDF does NOT make a general question a PDF question.
"Who was Homi J. Bhabha?" = WIKIPEDIA.
"What does my PDF say about source coding?" = PDF.

Question: {question}
"""
    try:
        response = llm.invoke(prompt)
        text = str(response.content).upper().strip()
        for route in ("MULTI", "WIKIPEDIA", "VISION", "WEB", "PDF"):
            if re.search(rf"\b{route}\b", text):
                return route
    except Exception as e:
        print(f"LLM router error: {e}")
    return "NONE"


def route_question(question: str, pdf_available: bool, image_available: bool) -> str:
    # Explicit user-source intent always wins.
    if is_document_question(question):
        return "PDF" if pdf_available else "NONE"

    if is_image_question(question):
        return "VISION" if image_available else "NONE"

    # Current information always goes to live web search.
    if is_current_question(question):
        return "WEB"

    route = llm_route(question)

    if route == "PDF":
        return "PDF" if pdf_available else "NONE"
    if route == "VISION":
        return "VISION" if image_available else "NONE"
    if route == "MULTI":
        if pdf_available and image_available:
            return "MULTI"
        if pdf_available:
            return "PDF"
        if image_available:
            return "VISION"
        return "NONE"
    if route == "WEB":
        return "WEB"
    if route == "WIKIPEDIA":
        return "WIKIPEDIA"

    # Safe default for ordinary non-current questions.
    return "WIKIPEDIA"


def execute_route(route: str, question: str, image_path: str | None) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []

    if route == "PDF":
        results.append(("search_documents", str(search_documents.invoke({"query": question}))))
        return results

    if route == "VISION":
        if not image_path:
            return [("describe_image", "No image is currently available.")]
        results.append(("describe_image", str(describe_image.invoke({
            "image_path": image_path,
            "question": question,
        }))))
        return results

    if route == "WEB":
        results.append(("web_search", str(web_search.invoke({"query": question}))))
        return results

    if route == "WIKIPEDIA":
        results.append(("wikipedia_search", str(wikipedia_search.invoke({"query": question}))))
        return results

    if route == "MULTI":
        if image_path:
            results.append(("describe_image", str(describe_image.invoke({
                "image_path": image_path,
                "question": question,
            }))))
        results.append(("search_documents", str(search_documents.invoke({"query": question}))))
        return results

    return results


def synthesize_answer(question: str, tool_results: list[tuple[str, str]]) -> str:
    if not tool_results:
        return "I could not retrieve information from a suitable source."

    formatted = []
    for tool_name, result in tool_results:
        result = str(result)
        if len(result) > 5000:
            result = result[:5000] + "\n[Result truncated]"
        formatted.append(f"SOURCE: {tool_name}\n\n{result}")

    prompt = f"""
Answer the user's question using the retrieved source information below.
Do not invent facts. Do not claim a source that was not used.
If the source says no information was found, state that honestly.

USER QUESTION:
{question}

RETRIEVED SOURCES:
{chr(10).join(formatted)}
"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Final answer generation failed: {e}"


def run_hybrid_agent(question: str, pdf_available: bool = False, image_path: str | None = None) -> str:
    image_available = image_path is not None

    route = route_question(
        question=question,
        pdf_available=pdf_available,
        image_available=image_available,
    )

    print("\n==============================")
    print("HYBRIDSIGHT ROUTE")
    print("==============================")
    print(f"Route           : {route}")
    print(f"PDF available   : {pdf_available}")
    print(f"Image available : {image_available}")

    if route == "NONE":
        if is_document_question(question):
            return "📄 No PDF is currently available. Please upload and index a PDF first."
        if is_image_question(question):
            return "🖼️ No image is currently available. Please upload an image first."
        return "I could not identify a suitable information source."

    tool_results = execute_route(
        route=route,
        question=question,
        image_path=image_path,
    )

    print("\n==============================")
    print("TOOLS EXECUTED")
    print("==============================")
    for tool_name, _ in tool_results:
        print(f"- {tool_name}")

    return synthesize_answer(question, tool_results)
