import streamlit as st
import json
from dotenv import load_dotenv
from tools import llm, tools
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import os
from langchain_tavily import TavilySearch

# -------------------------------
# Load environment
# -------------------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# -------------------------------
# Define Models and Tools
# -------------------------------
tavily_cfp = TavilySearch(
    max_results=20,
    search_depth="advanced",
    include_domains=["wikicfp.com"]
)
conference_tool = [tools[0], tavily_cfp]
tool_llm = llm.bind_tools(conference_tool)

class ConferenceInfo(BaseModel):
    conference_name: str = Field(..., description="Name of the conference.")
    location: Optional[str] = Field(None, description="Location of the conference.")
    date: Optional[str] = Field(None, description="Conference dates (start–end).")
    topics: Optional[str] = Field(None, description="Main topics covered.")
    submission_deadline: Optional[str] = Field(None, description="Paper submission deadline.")
    website: Optional[str] = Field(None, description="Conference website or CFP link.")

class ConferenceList(BaseModel):
    topic: str = Field(..., description="User’s research area.")
    conferences: List[ConferenceInfo] = Field(..., description="List of upcoming or relevant conferences.")

conference_llm = tool_llm.with_structured_output(ConferenceList)

# -------------------------------
# Backend function
# -------------------------------
def get_conferences(query: str):
    today = datetime.now().strftime("%Y-%m-%d")

    system_prompt = f"""
    You are an expert research assistant.
    Use the WikiCFP website (through the Tavily tool) to find real upcoming conferences related to the given topic.
    Focus only on *future* events (after {today}).
    For each event, extract:
    - conference name
    - location
    - dates
    - submission deadline
    - topics
    - website link (if available)
    Sort the conferences by their date in ascending order.
    Return a clean structured JSON following the provided schema.
    """

    messages = [
        ("system", system_prompt),
        ("human", f"Find conferences for: {query}")
    ]

    response = conference_llm.invoke(messages)

    while hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            tool_name = call["name"]
            args = call["args"]
            tool_fn = next(t for t in conference_tool if t.name == tool_name)
            tool_result = tool_fn.invoke(args)
            response = conference_llm.invoke([
                *messages,
                ("tool", f"Tool '{tool_name}' output: {tool_result}")
            ])

    if isinstance(response, ConferenceList):
        return response.dict()
    elif hasattr(response, "content"):
        return json.loads(response.content)
    else:
        return {"error": "Unexpected output", "raw": str(response)}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🎓 Conference Finder", layout="wide")

st.title("🎓 Research Conference Finder")
st.caption("Find upcoming academic conferences relevant to your topic using Arxiv + WikiCFP data.")

if "conference_data" not in st.session_state:
    st.session_state.conference_data = None

topic = st.text_input("Enter your research area or topic:")

if st.button("Find Conferences"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        with st.spinner("Searching for relevant conferences..."):
            data = get_conferences(topic)
            st.session_state.conference_data = data

# -------------------------------
# Display Results
# -------------------------------
if st.session_state.conference_data:
    data = st.session_state.conference_data

    if "error" in data:
        st.error(f"Error: {data['error']}")
        if "raw" in data:
            st.text(data["raw"])
    else:
        st.subheader(f"📘 Topic: {data['topic']}")
        st.markdown(f"**Total Conferences Found:** {len(data['conferences'])}")

        for i, conf in enumerate(data["conferences"], start=1):
            with st.expander(f"**{i}. {conf['conference_name']}**"):
                st.write(f"**Location:** {conf.get('location', 'N/A')}")
                st.write(f"**Dates:** {conf.get('date', 'N/A')}")
                st.write(f"**Submission Deadline:** {conf.get('submission_deadline', 'N/A')}")
                st.write(f"**Topics:** {conf.get('topics', 'N/A')}")
                if conf.get("website"):
                    st.markdown(f"[🔗 Conference Website]({conf['website']})")

        st.divider()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇ Download JSON Report",
            data=json_str,
            file_name=f"conferences_{topic.replace(' ', '_')}.json",
            mime="application/json"
        )
