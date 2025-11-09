import streamlit as st
import json
from dotenv import load_dotenv
from tools import llm, tools
from pydantic import BaseModel, Field
from typing import List, Optional
import os
from langchain_tavily import TavilySearch

# -------------------------------
# Load environment
# -------------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# -------------------------------
# Define models and tools
# -------------------------------
tavily_new = TavilySearch(max_results=10, search_depth="advanced")
review_tool = [tools[0], tavily_new]

class SimplePaperInfo(BaseModel):
    title: str = Field(..., description="Title of the paper.")
    authors: Optional[List[str]] = Field(None, description="Authors of the paper, if available.")
    year: Optional[int] = Field(None, description="Publication year, if known.")
    link: Optional[str] = Field(None, description="Link to the paper (arXiv or other).")
    abstract: Optional[str] = Field(None, description="Brief abstract or summary of the paper.")
    key_contribution: Optional[str] = Field(None, description="Main idea or contribution of the paper.")
    relevance: str = Field(..., description="Why this paper is relevant to the user’s topic.")

class LiteratureReview(BaseModel):
    topic: str = Field(..., description="The topic or query for the literature review.")
    papers: List[SimplePaperInfo] = Field(..., description="List of relevant papers for this topic.")
    summary: Optional[str] = Field(None, description="Overall summary or synthesis of findings across papers.")

tool_llm = llm.bind_tools(review_tool)
review_llm = tool_llm.with_structured_output(LiteratureReview)

# -------------------------------
# Backend function
# -------------------------------
def review_papers(user_query: str):
    system_prompt = """
    You are a literature review agent specialized in academic research.
    Use the tools (arxiv, tavily) to fetch and analyze papers related to the user's query.
    Retrieve factual information (titles, abstracts, methods, results, etc.)
    and return it strictly as a JSON object following the provided schema.
    If the user query is very broad, ask the user and try to narrow it down and focus on the most relevant aspects.
    """

    messages = [
        ("system", system_prompt),
        ("human", user_query),
    ]

    response = review_llm.invoke(messages)

    while hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            tool_name = call["name"]
            args = call["args"]
            tool_fn = next(t for t in review_tool if t.name == tool_name)
            tool_result = tool_fn.invoke(args)
            response = review_llm.invoke([
                *messages,
                ("tool", f"Tool '{tool_name}' output: {tool_result}")
            ])

    try:
        if isinstance(response, LiteratureReview):
            return response.dict()
        elif hasattr(response, "content"):
            return json.loads(response.content)
        elif isinstance(response, dict):
            return response
        else:
            return {"error": "Unexpected response format", "raw": str(response)}
    except Exception as e:
        return {"error": str(e), "raw": str(response)}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Literature Review Assistant", layout="wide")

st.title(" Literature Review Assistant")
st.caption("Automatically find and summarize academic papers for your research topic using Arxiv + Tavily tools.")

# Session state for caching results
if "review_data" not in st.session_state:
    st.session_state.review_data = None

topic = st.text_input("Enter your research topic or query:")

if st.button("Generate Literature Review"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.spinner("Fetching and analyzing papers..."):
            result = review_papers(topic)
            st.session_state.review_data = result

# Display Results
if st.session_state.review_data:
    data = st.session_state.review_data

    if "error" in data:
        st.error(f" Error: {data['error']}")
        if "raw" in data:
            st.text(data["raw"])
    else:
        st.subheader(f" Topic: {data['topic']}")
        if data.get("summary"):
            st.markdown(f"###  Summary\n{data['summary']}")

        st.markdown("###  Relevant Papers")
        for i, paper in enumerate(data["papers"], start=1):
            with st.expander(f"**{i}. {paper['title']}**"):
                st.write(f"**Authors:** {', '.join(paper.get('authors', [])) if paper.get('authors') else 'N/A'}")
                st.write(f"**Year:** {paper.get('year', 'N/A')}")
                st.write(f"**Key Contribution:** {paper.get('key_contribution', 'N/A')}")
                st.write(f"**Relevance:** {paper['relevance']}")
                if paper.get("abstract"):
                    st.markdown(f"**Abstract:**\n{paper['abstract']}")
                if paper.get("link"):
                    st.markdown(f"[🔗 View Paper]({paper['link']})")

        st.divider()
        json_str = json.dumps(data, indent=2)
        st.download_button(
            label="⬇ Download JSON Report",
            data=json_str,
            file_name=f"literature_review_{topic.replace(' ', '_')}.json",
            mime="application/json"
        )
