import streamlit as st
from dotenv import load_dotenv
from tools import llm, tools
from typing import Literal
from pydantic import BaseModel, Field
from prompt_library_2 import basic_prompt, COT_prompt, product_based_prompt, depth_research_prompt

load_dotenv()

class QueryLevelSchema(BaseModel):
    technique: Literal["Basic", "Chain-of-thought"] = Field(description="Prompting technique.")
    type: Literal["Product_Based", "Depth_Research"] = Field(description="Type of ideation task.")

Agent = llm.with_structured_output(QueryLevelSchema)

def query_level(user_query: str):
    convo_messages = [
        ("system", """
        You are an intelligent routing agent for an ideation assistant.
        Choose which reasoning style best fits the user query:
        - 'Basic' for simple, unclear, or curiosity-driven questions.
        - 'Chain-of-thought' for analytical or multi-step reasoning.
        - 'Product_Based' for creative, innovation-oriented ideas.
        - 'Depth_Research' for complex, exploratory, or academic ideation.
        If a user asks a vague incomplete query, assume that it is in the direction of ideation.
        For example: "Medical Imaging project idea" → return clear project ideas, directions, and possible scopes.
        """),
        ("human", user_query),
    ]
    return Agent.invoke(convo_messages)

def prompt(level):
    prompts = []
    if level.technique == "Basic":
        prompts.append(basic_prompt)
    elif level.technique == "Chain-of-thought":
        prompts.append(COT_prompt)
    if level.type == "Product_Based":
        prompts.append(product_based_prompt)
    elif level.type == "Depth_Research":
        prompts.append(depth_research_prompt)
    return prompts

ideation_tool = tools[1:3]
ideation_llm = llm.bind_tools(ideation_tool)

def run_ideation_chat(user_query, conversation):
    level = query_level(user_query)
    prompts = prompt(level)
    final_prompt = "  ".join(prompts)

    messages = [
        ("system", final_prompt),
        *conversation,
        ("human", user_query),
    ]

    response = ideation_llm.invoke(messages)
    if hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            tool_name = call["name"]
            args = call["args"]
            tool_fn = next(t for t in ideation_tool if t.name == tool_name)
            tool_result = tool_fn.invoke(args)
            response = ideation_llm.invoke([
                *messages,
                ("tool", f"Tool '{tool_name}' output: {tool_result}")
            ])

    ans = response.content if hasattr(response, "content") else str(response)
    if isinstance(ans, list):
        ans = ans[0].get("text", str(ans))
    return ans


# --- Streamlit UI ---
st.set_page_config(page_title="Ideation Assistant", page_icon="💡", layout="wide")

st.title("💡 Ideation Assistant")
st.caption("Your creative partner for brainstorming project ideas and innovation directions.")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display all past messages
for role, msg in st.session_state.conversation:
    with st.chat_message("user" if role == "human" else "assistant"):
        st.markdown(msg)

# User input
user_query = st.chat_input("Type your idea, question, or topic...")

if user_query:
    st.session_state.conversation.append(("human", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans = run_ideation_chat(user_query, st.session_state.conversation)
            st.markdown(ans)

    st.session_state.conversation.append(("assistant", ans))
