from dotenv import load_dotenv
from tools import llm, tools 
from typing import Literal , Annotated,TypedDict
from pydantic import BaseModel, Field 
from typing import List, Optional
from datetime import datetime
import json
import os 

from langchain_tavily import TavilySearch

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

tavily_cfp = TavilySearch(
    max_results=20,
    search_depth="advanced",
    include_domains=["wikicfp.com"]
)
conference_tool = [tools[0], tavily_cfp]  # Updating tools to include Arxiv and Tavily for conference

tool_llm = llm.bind_tools(conference_tool)
# class ConferenceSchema(BaseModel):
#     conference_name : str = Field(description="Name of the conference.")
#     location : str = Field(description="Location where the conference is being held.")
#     date : str = Field(description="Date of the conference.")
#     topics_covered : str = Field(description="Topics that will be covered in the conference.")
#     submission_deadline : str = Field(description="Submission deadline for papers or abstracts.")
#     website : str = Field(description="Official website of the conference.")


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
            print(f" Tool invoked: {tool_name} | Args: {args}")
            tool_fn = next(t for t in conference_tool if t.name == tool_name)
            tool_result = tool_fn.invoke(args)
            print(f" Tool result snippet: {str(tool_result)[:250]}...")
            response = conference_llm.invoke([
                *messages,
                ("tool", f"Tool '{tool_name}' output: {tool_result}")
            ])

    if isinstance(response, ConferenceList):
        data = response.dict()
        print(f"\nTotal conferences found: {len(data['conferences'])}")

        filename = f"conferences_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = os.path.join(os.getcwd(), filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nSaved results to: {output_path}")
        print("\nPreview:\n")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        return data
    else:
        print("Unexpected output:", response)
        return response



if __name__ == "__main__":
    user_topic = input("Enter your research topic to find relevant conferences: ")
    get_conferences(user_topic)