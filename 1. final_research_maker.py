#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install -U langchain langgraph duckduckgo-search gradio openai tiktoken')


# In[3]:


from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph
from typing import TypedDict, List

print("Imports OK")

# =====================
# State Definition
# =====================

class AgentState(TypedDict):
    topic: str
    plan: str
    research: List[str]
    report: str


# =====================
# Agents
# =====================

def planner_agent(state: AgentState) -> AgentState:
    topic = state["topic"]

    plan = f"""
1. Search basic information about {topic}
2. Identify key concepts
3. Collect important points
4. Summarize findings
"""

    state["plan"] = plan
    return state


search_tool = DuckDuckGoSearchRun()

def research_agent(state: AgentState) -> AgentState:
    topic = state["topic"]
    results = search_tool.run(topic)
    state["research"] = [results]
    return state


def writer_agent(state: AgentState) -> AgentState:
    research = state["research"][0]

    report = f"""
=== Research Report ===

Topic: {state['topic']}

Key Findings:
{research}

Conclusion:
This report summarizes publicly available information.
"""

    state["report"] = report
    return state


# =====================
# LangGraph
# =====================

builder = StateGraph(AgentState)

builder.add_node("planner", planner_agent)
builder.add_node("research", research_agent)
builder.add_node("writer", writer_agent)

builder.set_entry_point("planner")
builder.add_edge("planner", "research")
builder.add_edge("research", "writer")

graph = builder.compile()

print("Graph ready!")


# In[5]:


initial_state = {
    "topic": "Agentic AI in Healthcare",
    "plan": "",
    "research": [],
    "report": ""
}

result = graph.invoke(initial_state)

print("\n=== FINAL REPORT ===\n")
print(result["report"])


# In[6]:


# Visualize LangGraph as Mermaid diagram

graph_mermaid = graph.get_graph().draw_mermaid()

print(graph_mermaid)


# In[9]:


from IPython.display import Markdown, display

display(Markdown(f"```mermaid\n{graph_mermaid}\n```"))


# In[13]:


import gradio as gr
import time, json, os, uuid
from datetime import datetime

CHAT_FILE = "chats.json"

# ---------------- STORAGE ----------------

def load_chats():
    if not os.path.exists(CHAT_FILE):
        return []
    with open(CHAT_FILE,"r",encoding="utf8") as f:
        return json.load(f)

def save_chats(data):
    with open(CHAT_FILE,"w",encoding="utf8") as f:
        json.dump(data,f,indent=2)

def sidebar_labels():
    return [f"{c['id']} | {c['topic']}" for c in load_chats()]

# ---------------- CHAT OPS ----------------

def open_chat(label):
    if not label:
        return ""
    cid = label.split("|")[0].strip()
    for c in load_chats():
        if c["id"] == cid:
            return c["report"]
    return ""

def new_chat():
    return "", ""

def delete_chat(label):
    if not label:
        return gr.Radio.update(choices=[]), ""

    cid = label.split("|")[0].strip()
    chats = [c for c in load_chats() if c["id"] != cid]
    save_chats(chats)

    return gr.Radio.update(choices=sidebar_labels(), value=None), ""

# ---------------- STREAMING AGENT ----------------

def run_agent_stream(topic):

    state = {
        "topic": topic,
        "plan": "",
        "research": [],
        "report": ""
    }

    text = "🧠 Planner running...\n"
    yield text
    time.sleep(.5)

    text += "🔍 Research agent running...\n"
    yield text
    time.sleep(.5)

    text += "✍ Writer agent running...\n\n"
    yield text

    result = graph.invoke(state)
    report = result["report"]

    for ch in report:
        text += ch
        yield text
        time.sleep(.01)

    chats = load_chats()

    chats.append({
        "id": str(uuid.uuid4())[:8],
        "topic": topic,
        "report": report,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    save_chats(chats)

# ---------------- UI ----------------

with gr.Blocks() as demo:

    gr.Markdown("# 🧠 AutoScientist")

    with gr.Row():

        with gr.Column(scale=1):
            gr.Markdown("### History")

            sidebar = gr.Radio(choices=sidebar_labels())

            new_btn = gr.Button("➕ New Chat")
            del_btn = gr.Button("🗑 Delete")

        with gr.Column(scale=3):
            topic = gr.Textbox(label="Enter Topic")
            output = gr.Textbox(lines=22,label="Live Output")
            run = gr.Button("Run")

    sidebar.change(open_chat, sidebar, output)
    new_btn.click(new_chat,None,[topic,output])
    del_btn.click(delete_chat,sidebar,[sidebar,output])

    run.click(run_agent_stream,topic,output)
    run.click(lambda: gr.Radio.update(choices=sidebar_labels()),None,sidebar)

demo.launch()


# In[ ]:




