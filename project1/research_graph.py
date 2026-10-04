"""Multi-agent research graph: Planner -> parallel Researchers -> Synthesizer -> Reflector.

Same code as project1.ipynb, as a module the FastAPI backend can import.
"""
import operator
import os
import re
from pathlib import Path
from typing import Annotated, List, Literal

import wikipedia
from ddgs import DDGS
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Send

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")   # GROQ_API_KEY, HF_TOKEN, ...


# ───────────────────────────── models ─────────────────────────────
# max_new_tokens: the default (512) gets used up by Qwen3's thinking -> empty answers
endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen3-32B", task="text-generation", max_new_tokens=4096)
llm = ChatHuggingFace(llm=endpoint)     # researchers' tool-calling model
llm2 = ChatGroq(model="openai/gpt-oss-120b")   # planning, grading, writing, reviewing


# ───────────────────────────── vector store ─────────────────────────────
PDFS = [
    "1706.03762v7.pdf",   # Attention Is All You Need (Transformer)
    "2005.11401v4.pdf",   # Retrieval-Augmented Generation (RAG)
    "2210.03629v3.pdf",   # ReAct
    "2310.11511v1.pdf",   # Self-RAG
    "2401.15884v3.pdf",   # Corrective RAG (CRAG)
]

embeddings = HuggingFaceEmbeddings()

# saved to disk: the papers are embedded once, later starts just load the store
# (this also avoids the duplicate chunks you get from re-running from_documents in memory)
store = Chroma(collection_name="papers", embedding_function=embeddings, persist_directory=str(HERE / "chroma_db"))
if not store.get(limit=1)["ids"]:
    docs = [page for pdf in PDFS for page in PyPDFLoader(str(HERE / pdf)).load()]
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(PDFS)} papers")
retriever = store.as_retriever()


# ───────────────────────────── tools ─────────────────────────────
@tool
def paper_search_tool(query: str) -> str:
    """Search a vector store of 5 AI research papers: 'Attention Is All You Need' (Transformer),
    'Retrieval-Augmented Generation (RAG)', 'ReAct', 'Self-RAG', and 'Corrective RAG (CRAG)'.
    Use it for questions about transformers, attention, retrieval-augmented generation,
    or reasoning-and-acting agents.
    Input: a specific search query in plain English (e.g. 'why is attention scaled by sqrt(d_k)').
    Returns: the most relevant passages, each with its paper and page number."""
    results = retriever.invoke(query)
    return "\n\n".join(
        f"[{i}] {os.path.basename(d.metadata.get('source', ''))}, page {d.metadata.get('page_label', '?')}\n{d.page_content}"
        for i, d in enumerate(results, start=1)
    )


# Wikipedia rate-limits the wikipedia package's default user agent (429 -> JSONDecodeError)
wikipedia.set_user_agent("agentic-ai-revision/1.0 (student project)")
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)


@tool
def wiki_search_tool(query: str) -> str:
    """Search Wikipedia for general knowledge: people, places, organizations, historical events,
    and definitions of concepts.
    Use it for questions that are NOT covered by the 5 research papers in the vector store,
    or to get general background on a topic (e.g. 'who is Geoffrey Hinton', 'what is BERT').
    Do NOT use it for detailed questions about the Transformer, RAG, ReAct, Self-RAG,
    or CRAG papers; use paper_search_tool for those.
    Input: a short search term or question in plain English (e.g. 'Cristiano Ronaldo').
    Returns: short summaries of the top matching Wikipedia pages."""
    try:
        results = wikipedia_wrapper.load(query)
    except Exception as e:   # rate limit / block returns HTML, not JSON
        return f"Wikipedia search failed: {e}"
    return "\n\n".join(
        f"[{i}] {d.metadata.get('title', '')} ({d.metadata.get('source', '')})\n{d.page_content}"
        for i, d in enumerate(results, start=1)
    ) or "No Wikipedia results."


@tool
def duck_duck_search(query: str) -> str:
    """Search the web with DuckDuckGo for recent, current, or very specific information.
    Use it for news, recent events, latest releases or versions, current facts that may have
    changed (e.g. 'latest GPT model', 'which AI models use transformers today'),
    or anything not found by the other tools.
    Do NOT use it for questions about the 5 research papers (use paper_search_tool)
    or for stable general background like biographies and definitions (use wiki_search_tool).
    Input: a concise search query in plain English, as you would type into a search engine.
    Returns: the top web results, each with a title, URL, and short snippet."""
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"
    return "\n\n".join(
        f"[{i}] {r['title']} ({r['href']})\n{r['body']}"
        for i, r in enumerate(results, start=1)
    ) or "No web results."


tools = [paper_search_tool, wiki_search_tool, duck_duck_search]
llm_with_tools = llm.bind_tools(tools)


# ───────────────────────────── state ─────────────────────────────
class ResearcherState(BaseModel):
    sub_question: str                                          # the ONE sub-question this researcher owns
    messages: Annotated[List[AnyMessage], add_messages] = []   # private agent <-> tools conversation
    evidence: Annotated[List[dict], operator.add] = []         # results the grader kept
    rounds: int = 0                                            # how many tool rounds have been graded


class ResearcherOutput(BaseModel):
    evidence: Annotated[List[dict], operator.add] = []         # the ONLY thing sent back to the main graph


class MainState(BaseModel):
    questions: Annotated[List[AnyMessage], add_messages]
    sub_questions: Annotated[List[str], operator.add] = []     # planner's + reflector's follow-ups
    evidence: Annotated[List[dict], operator.add] = []         # merged from all researchers
    answers: Annotated[List[AnyMessage], add_messages] = []    # every draft; answers[-1] is the latest
    decision: str = ""                                         # reflector: accept / revise / research_more
    feedback: str = ""                                         # reflector -> synthesizer (revise)
    follow_ups: List[str] = []                                 # reflector -> new researchers (research_more)
    reflections: int = 0                                       # how many times the reflector has reviewed


# ───────────────────────────── planner ─────────────────────────────
class Planner(BaseModel):
    sub_questions: list[str] = Field(
        description="The user's question broken into 1-4 short, self-contained sub-questions, "
                    "each answerable with a single search. Use just 1 if the question is simple."
    )


PLANNER_PROMPT = SystemMessage(content="""You are the Planner in a research team.
Break the user's question into sub-questions. Each sub-question goes to a separate researcher who works in parallel and does NOT see the original question.

Rules:
- Write 1 to 4 sub-questions. Use just 1 if the question is simple.
- Make each one self-contained: repeat names and terms instead of writing "it", "this" or "they".
- Make each one cover a different part of the question, with no overlap.
- Together, they must cover the whole question.
- Do not answer them.""")

planner_llm = llm2.with_structured_output(Planner)


def planner_node(state: MainState) -> dict:
    plan = planner_llm.invoke([PLANNER_PROMPT, state.questions[-1]])
    return {"sub_questions": plan.sub_questions}


# ───────────────────────────── researcher ─────────────────────────────
RESEARCHER_PROMPT = """You are a Researcher in a multi-agent research team. Your job is to gather evidence for ONE sub-question, not to write the final answer.

## The team
- Planner: split the user's question into sub-questions. Other researchers are working on the other ones in parallel.
- Researcher (you): finds evidence for your sub-question using your tools.
- Grader: reviews each tool result before you see it. It removes irrelevant results and adds a note starting with "GRADER:" that says what is still missing.
- Synthesizer: receives the evidence from all researchers and writes the final answer.
- Reflector: reviews the final answer and may send follow-up sub-questions to new researchers.

## Your sub-question
{sub_question}

## Your tools
1. paper_search_tool: searches 5 AI research papers (Transformer, RAG, ReAct, Self-RAG, CRAG). Use it for technical details from the papers.
2. wiki_search_tool: searches Wikipedia. Use it for general background, definitions, people, and history.
3. duck_duck_search: searches the web. Use it for recent or current information, and for anything the other tools couldn't find.

## How to work
1. Pick the tool that fits your sub-question best. Use more than one if the sub-question has several parts.
2. Read the GRADER notes. If one says "nothing relevant" or that something is still missing, rewrite the query with different keywords or switch to another tool.
3. Use specific search queries. Do not answer from your own memory.
4. Stop calling tools once you have enough evidence, or once you have tried every suitable tool.

## When you are done
Do not write the final answer. Reply with one line: "covered", or "not found: <what you tried>"."""


def research_agent_node(state: ResearcherState) -> dict:
    sys_msg = SystemMessage(content=RESEARCHER_PROMPT.format(sub_question=state.sub_question))
    response = llm_with_tools.invoke([sys_msg, HumanMessage(state.sub_question)] + state.messages)
    return {"messages": [response]}


class GraderOutput(BaseModel):
    relevant: list[int] = Field(description="Numbers of the results that help answer the question")
    still_missing: str = Field(description="One sentence on what is still missing, or empty if nothing")


GRADER_PROMPT = SystemMessage(content="""You are a relevance grader for a research assistant.
You will receive a question and a numbered list of search results.

1. Return the numbers of the results that directly help answer the question, even partially. Ignore results that are only loosely related.
2. In one sentence, say what information is still missing. Leave it empty if nothing is missing.

If no result is relevant, return an empty list. Do not answer the question yourself.""")

grader_llm = llm2.with_structured_output(GraderOutput)


def research_grader_node(state: ResearcherState) -> dict:
    question = state.sub_question

    # this round's tool results = the ToolMessages at the end of the conversation
    tool_msgs = []
    for m in reversed(state.messages):
        if not isinstance(m, ToolMessage):
            break
        tool_msgs.append(m)

    cleaned, evidence = [], []
    for m in tool_msgs:
        # split the message into its numbered results [1], [2], ... and grade each one
        results = re.split(r"\n\n(?=\[\d+\])", m.content.strip())
        numbered = "\n\n".join(f"Result {i}:\n{r}" for i, r in enumerate(results, start=1))
        out = grader_llm.invoke([GRADER_PROMPT, HumanMessage(f"Question: {question}\n\n{numbered}")])

        kept = [results[i - 1] for i in out.relevant if 1 <= i <= len(results)]
        evidence += [{"sub_question": question, "tool": m.name, "text": r} for r in kept]

        if kept:
            note = f"GRADER: kept {len(kept)} of {len(results)}. Still missing: {out.still_missing or 'nothing'}"
        else:
            note = "GRADER: nothing relevant. Rewrite the query with different keywords or try another tool."

        # same id -> add_messages replaces the original ToolMessage instead of appending a new one
        cleaned.append(m.model_copy(update={"content": "\n\n".join(kept + [note])}))

    return {"messages": cleaned, "evidence": evidence, "rounds": state.rounds + 1}


MAX_ROUNDS = 3   # max tool rounds per researcher


def researcher_route(state: ResearcherState):
    if state.rounds >= MAX_ROUNDS:        # stop runaway search loops
        return END
    return tools_condition(state)         # tool calls -> "tools", otherwise -> END


rb = StateGraph(ResearcherState, output_schema=ResearcherOutput)
rb.add_node("agent", research_agent_node)
rb.add_node("tools", ToolNode(tools))
rb.add_node("grader", research_grader_node)
rb.add_edge(START, "agent")
rb.add_conditional_edges("agent", researcher_route, {"tools": "tools", END: END})
rb.add_edge("tools", "grader")
rb.add_edge("grader", "agent")
researcher = rb.compile()


# ───────────────────────────── fan-out ─────────────────────────────
def send_to_researchers(state: MainState) -> list[Send]:
    # the name "researcher" must match add_node("researcher", ...) below
    return [Send("researcher", {"sub_question": q}) for q in state.sub_questions]


# ───────────────────────────── synthesizer ─────────────────────────────
def strip_number(text: str) -> str:
    """'[3] paper.pdf, page 2\n...' -> 'paper.pdf, page 2\n...'"""
    return re.sub(r"^\[\d+\]\s*", "", text)


def unique_sources(evidence: list[dict]) -> list[dict]:
    """Drop duplicate results (the same passage can be found twice, with a different [n])."""
    seen, sources = set(), []
    for e in evidence:
        key = strip_number(e["text"])
        if key not in seen:
            seen.add(key)
            sources.append(e)
    return sources


def format_sources(sources: list[dict]) -> str:
    """One global numbering [1] ... [n], tagged with the sub-question each source answers."""
    return "\n\n".join(
        f"[{n}] (for: {s['sub_question']})\n{strip_number(s['text'])}"
        for n, s in enumerate(sources, start=1)
    )


def build_references(answer: str, sources: list[dict]) -> str:
    """References list for the [n] numbers the answer actually cites."""
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(sources)})
    if not cited:
        return ""
    lines = [f"[{n}] {strip_number(sources[n - 1]['text']).splitlines()[0]}" for n in cited]
    return "\n\n## References\n" + "\n".join(lines)


SYNTHESIZER_PROMPT = SystemMessage(content="""You are the Synthesizer in a research team. Researchers have gathered numbered sources. Write the final answer to the user's question using ONLY these sources.

Rules:
- Base every statement on the sources. Do not use outside knowledge.
- Cite every fact inline with its source number, like [1] or [2][5]. Only use numbers from the list.
- Cover every part of the question. If the sources don't answer a part, say so honestly.
- If sources disagree, say so and explain the difference.
- Answer the question directly first, then add supporting details. Be concise.
- Do NOT write a references list; it is added automatically.""")


def synthesizer_node(state: MainState) -> dict:
    sources = unique_sources(state.evidence)
    request = f"Question: {state.questions[-1].content}\n\nSources:\n{format_sources(sources)}"

    if state.feedback:   # revise loop: show the previous draft and what to fix
        request += (f"\n\nYour previous draft:\n{state.answers[-1].content}"
                    f"\n\nThe Reflector asked for these fixes:\n{state.feedback}"
                    "\n\nWrite the corrected answer.")

    answer = llm2.invoke([SYNTHESIZER_PROMPT, HumanMessage(request)]).content
    return {"answers": [AIMessage(answer + build_references(answer, sources))]}


# ───────────────────────────── reflector ─────────────────────────────
class Reflection(BaseModel):
    decision: Literal["accept", "revise", "research_more"] = Field(
        description="accept = supported and complete; revise = fix the writing; research_more = information is missing")
    feedback: str = Field(
        description="For 'revise': which claims are unsupported or wrongly cited, and how to fix them. Empty otherwise.")
    follow_up_questions: list[str] = Field(
        description="For 'research_more': 1-3 self-contained sub-questions covering only what is missing. Empty otherwise.")


REFLECTOR_PROMPT = SystemMessage(content="""You are the Reflector in a research team. You review a draft answer against the numbered sources it was written from.

Check, in this order:
1. Support: every fact must have a citation [n], and source n must actually say it. Facts with no citation, or with a citation that doesn't back them, are unsupported.
2. Coverage: the answer must cover every part of the user's question.

Then decide:
- accept: every fact is supported and every part of the question is covered. Minor style issues are fine.
- revise: the sources contain what is needed, but the answer has unsupported claims, wrong citations, or leaves out something the sources do say. Give specific feedback.
- research_more: the sources do NOT contain information needed for part of the question. Write 1-3 self-contained follow-up sub-questions for only the missing parts (repeat names instead of "it" or "this").

Be strict about made-up facts and lenient about style.""")

reflector_llm = llm2.with_structured_output(Reflection)


def reflector_node(state: MainState) -> dict:
    sources = unique_sources(state.evidence)
    review = reflector_llm.invoke([REFLECTOR_PROMPT, HumanMessage(
        f"Question: {state.questions[-1].content}\n\n"
        f"Sources:\n{format_sources(sources)}\n\n"
        f"Draft answer:\n{state.answers[-1].content}"
    )])

    follow_ups = review.follow_up_questions if review.decision == "research_more" else []
    return {
        "decision": review.decision,
        "feedback": review.feedback if review.decision == "revise" else "",
        "follow_ups": follow_ups,
        "sub_questions": follow_ups,          # keep a record of every sub-question asked
        "reflections": state.reflections + 1,
    }


MAX_LOOPS = 2   # max times the reflector can send work back


def route_after_reflector(state: MainState):
    if state.decision == "accept" or state.reflections > MAX_LOOPS:
        return END
    if state.decision == "revise":
        return "synthesizer"
    if state.follow_ups:                  # research_more -> new researchers, in parallel
        return [Send("researcher", {"sub_question": q}) for q in state.follow_ups]
    return END                            # research_more but no follow-ups -> nothing to do


# ───────────────────────────── full graph ─────────────────────────────
builder = StateGraph(MainState)
builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher)        # the compiled subgraph is used as a node
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("reflector", reflector_node)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", send_to_researchers, ["researcher"])
builder.add_edge("researcher", "synthesizer")     # waits for ALL parallel researchers to finish
builder.add_edge("synthesizer", "reflector")
builder.add_conditional_edges("reflector", route_after_reflector, ["synthesizer", "researcher", END])

research_graph = builder.compile()
