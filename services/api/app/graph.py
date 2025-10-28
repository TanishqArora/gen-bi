import os
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOllama
from .tools import query, ensure_collection


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.2)
ensure_collection()


# --- Simple state ---
class State(Dict[str, Any]):
    pass




def retrieve_node(state: State) -> State:
    q = state.get("question", "")
    retrieved = query(q, k=5)
    state["contexts"] = retrieved
    return state




def answer_node(state: State) -> State:
    sys = (
    "You are a precise, terse assistant. Use the provided context to answer. "
    "Cite sources as [source] inline when used. If unsure, say you don't know."
    )
    ctx_lines = [f"- {d['text']} [source: {d['source']}]" for d in state.get("contexts", [])]
    ctx = "\n".join(ctx_lines) or "(no context)"


    content = f"""
    [CONTEXT]\n{ctx}\n\n[QUESTION]\n{state.get('question','')}
    """


    resp = llm.invoke([
    {"role": "system", "content": sys},
    {"role": "user", "content": content},
    ])
    state["answer"] = resp.content
    return state




# --- Graph ---
wf = StateGraph(State)
wf.add_node("retrieve", retrieve_node)
wf.add_node("answer", answer_node)
wf.add_edge(START, "retrieve")
wf.add_edge("retrieve", "answer")
wf.add_edge("answer", END)


app_graph = wf.compile()