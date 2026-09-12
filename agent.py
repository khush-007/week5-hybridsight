"""
HybridSight Agent

Architecture:

User Question
      ↓
Planner LLM
      ↓
Select required tools
      ↓
Execute tools
      ↓
Final LLM synthesis

Supported tools:
- PDF / RAG
- Vision
- DuckDuckGo
- Wikipedia
"""

from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from config import get_llm
from tools_rag import search_documents
from tools_vision import describe_image


# =========================================================
# MODELS
# =========================================================

llm = get_llm()


# =========================================================
# RAW TOOLS
# =========================================================

duckduckgo = DuckDuckGoSearchRun()

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)


# =========================================================
# SAFE WEB TOOL
# =========================================================

@tool
def web_search(query: str) -> str:
    """
    Search the live web for current or recent information.

    Use this for:
    - latest information
    - current events
    - recent news
    - information that changes over time
    """

    try:
        result = duckduckgo.invoke(query)

        if not result:
            return "No useful web results found."

        return str(result)

    except Exception as e:
        return f"Web search failed: {e}"


# =========================================================
# SAFE WIKIPEDIA TOOL
# =========================================================

@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for general encyclopedic knowledge.

    Use this for:
    - concepts
    - technologies
    - history
    - people
    - places
    """

    try:
        result = wikipedia.invoke(query)

        if not result:
            return "No useful Wikipedia results found."

        return str(result)

    except Exception as e:
        return f"Wikipedia search failed: {e}"


# =========================================================
# PLANNER SCHEMA
# =========================================================

class ToolTask(BaseModel):

    tool: Literal[
        "search_documents",
        "describe_image",
        "web_search",
        "wikipedia_search",
    ] = Field(
        description="The tool required for this task."
    )

    query: str = Field(
        description=(
            "The specific sub-question or query that "
            "should be sent to the selected tool."
        )
    )


class ToolPlan(BaseModel):

    tasks: list[ToolTask] = Field(
        description=(
            "List of tool tasks required to answer "
            "the user's question. Include multiple "
            "tasks when multiple sources are needed."
        )
    )


# =========================================================
# PLANNER
# =========================================================

planner = llm.with_structured_output(ToolPlan)


def create_plan(
    question: str,
    pdf_available: bool,
    image_available: bool,
) -> ToolPlan:

    planner_prompt = f"""
You are the planning component of HybridSight.

Your job is to understand the user's question and
decide which information sources are required.

AVAILABLE SOURCES:

PDF/document available:
{pdf_available}

Image available:
{image_available}

AVAILABLE TOOLS:

1. search_documents
   Searches the user's uploaded PDF/document.

2. describe_image
   Analyzes the uploaded image.

3. web_search
   Searches the live internet for current/recent information.

4. wikipedia_search
   Searches Wikipedia for general encyclopedic knowledge.

IMPORTANT RULES:

- Understand the meaning of the user's question.
- Do NOT use keyword matching.
- Break a complex question into separate information needs.
- A question can require multiple tools.
- If PDF and image information are both required,
  create BOTH tasks.
- If current information is required, use web_search.
- Use wikipedia_search for general encyclopedic knowledge.
- Do not select tools that are unnecessary.
- If a source is unavailable, do not select its tool.
- Queries should be focused sub-questions that help answer
  the original question.

USER QUESTION:

{question}
"""

    try:

        plan = planner.invoke(planner_prompt)

        return plan

    except Exception as e:

        print("Planner error:", e)

        return ToolPlan(tasks=[])


# =========================================================
# TOOL EXECUTION
# =========================================================

def execute_task(
    task: ToolTask,
    image_path: str | None = None,
) -> str:

    try:

        # ---------------------------------------------
        # PDF
        # ---------------------------------------------

        if task.tool == "search_documents":

            return search_documents.invoke(
                {
                    "query": task.query
                }
            )

        # ---------------------------------------------
        # IMAGE
        # ---------------------------------------------

        if task.tool == "describe_image":

            if not image_path:

                return (
                    "No image is available for "
                    "vision analysis."
                )

            return describe_image.invoke(
                {
                    "image_path": image_path,
                    "question": task.query,
                }
            )

        # ---------------------------------------------
        # WEB
        # ---------------------------------------------

        if task.tool == "web_search":

            return web_search.invoke(
                {
                    "query": task.query
                }
            )

        # ---------------------------------------------
        # WIKIPEDIA
        # ---------------------------------------------

        if task.tool == "wikipedia_search":

            return wikipedia_search.invoke(
                {
                    "query": task.query
                }
            )

        return "Unknown tool requested."

    except Exception as e:

        return (
            f"Tool '{task.tool}' failed: {e}"
        )


# =========================================================
# FINAL SYNTHESIS
# =========================================================

def synthesize_answer(
    question: str,
    tool_results: list[tuple[str, str]],
) -> str:

    if not tool_results:

        return (
            "I could not identify a suitable information "
            "source to answer this question."
        )

    formatted_results = []

    for tool_name, result in tool_results:

        # Prevent unnecessarily huge context
        result = str(result)

        if len(result) > 5000:
            result = result[:5000] + "\n[Result truncated]"

        formatted_results.append(
            f"""
SOURCE: {tool_name}

{result}
"""
        )

    combined_sources = "\n".join(
        formatted_results
    )

    synthesis_prompt = f"""
You are HybridSight's final answer generator.

USER QUESTION:

{question}

INFORMATION RETRIEVED FROM TOOLS:

{combined_sources}

TASK:

Answer the user's original question using the
retrieved information.

IMPORTANT:

1. Combine information from ALL relevant sources.
2. Do not ignore a source just because another source
   contains more information.
3. If the question requires comparison, explicitly compare
   the relevant sources.
4. Do not invent information.
5. If the sources disagree, clearly mention the disagreement.
6. If information is missing, say that it is unavailable.
7. Give a clear, direct final answer.
8. Do not describe your internal reasoning.
"""

    try:

        response = llm.invoke(
            synthesis_prompt
        )

        return response.content

    except Exception as e:

        return (
            f"Final answer generation failed: {e}"
        )


# =========================================================
# MAIN HYBRID AGENT
# =========================================================

def run_hybrid_agent(
    question: str,
    pdf_available: bool = False,
    image_path: str | None = None,
) -> str:

    image_available = image_path is not None

    # -----------------------------------------------------
    # 1. PLAN
    # -----------------------------------------------------

    plan = create_plan(
        question=question,
        pdf_available=pdf_available,
        image_available=image_available,
    )

    print("\n==============================")
    print("HYBRIDSIGHT PLAN")
    print("==============================")

    for task in plan.tasks:

        print(
            f"- {task.tool}: {task.query}"
        )

    # -----------------------------------------------------
    # 2. EXECUTE
    # -----------------------------------------------------

    tool_results = []

    for task in plan.tasks:

        result = execute_task(
            task=task,
            image_path=image_path,
        )

        tool_results.append(
            (
                task.tool,
                result,
            )
        )

    # -----------------------------------------------------
    # 3. SYNTHESIZE
    # -----------------------------------------------------

    final_answer = synthesize_answer(
        question=question,
        tool_results=tool_results,
    )

    return final_answer