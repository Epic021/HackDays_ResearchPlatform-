from dotenv import load_dotenv
from tools import llm, tools 
from typing import Literal , Annotated,TypedDict
from pydantic import BaseModel, Field 
from typing import List, Optional
import os 
import json 

from langchain_tavily import TavilySearch

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

tavily_new = TavilySearch(max_results=10, search_depth="advanced")
review_tool = [tools[0], tavily_new]  # Updating tools to include Arxiv and Tavily for review



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
    

# structured_llm = llm.with_structured_output(json_schema)
# review_llm = structured_llm.bind_tools(review_tool)

tool_llm = llm.bind_tools(review_tool)
review_llm = tool_llm.with_structured_output(LiteratureReview)




####################################################################################


def review_papers(user_query: str):
    """
    Handles the entire LLM → tool → structured output process.
    """

    system_prompt = """
    You are a literature review agent specialized in academic research.
    Use the tools (arxiv, tavily) to fetch and analyze papers related to the user's query.
    Retrieve factual information (titles, abstracts, methods, results, etc.)
    and return it strictly as a JSON object following the provided schema.
    """

    messages = [
        ("system", system_prompt),
        ("human", user_query),
    ]

    # Step 1: LLM initial reasoning + tool usage
    response = review_llm.invoke(messages)

    # Step 2: If tool calls are generated, execute and feed results back
    while hasattr(response, "tool_calls") and response.tool_calls:
        for call in response.tool_calls:
            tool_name = call["name"]
            args = call["args"]

            print(f" Model invoked tool: {tool_name} | Args: {args}")

            # Find matching tool
            tool_fn = next(t for t in review_tool if t.name == tool_name)
            tool_result = tool_fn.invoke(args)

            print(f" Tool result snippet: {str(tool_result)[:300]}...")

            # Feed tool result back into the conversation
            response = review_llm.invoke([
                *messages,
                ("tool", f"Tool '{tool_name}' output: {tool_result}")
            ])

    # Step 3: Final structured result
    # Step 3: Handle structured result
    try:
        # If it's already a Pydantic object
        if isinstance(response, LiteratureReview):
            data = response.dict()
            print(json.dumps(data, indent=2))
            return data

        # If it’s a JSON string inside response.content
        elif hasattr(response, "content"):
            data = json.loads(response.content)
            print(json.dumps(data, indent=2))
            return data

        # If it’s a dict already
        elif isinstance(response, dict):
            print(json.dumps(response, indent=2))
            return response

        else:
            print(" Unexpected response type:", type(response))
            print(response)
            return response

    except Exception as e:
        print("Error while parsing response:", e)
        print("Raw output:")
        print(response)
        return response



# -------------------------------
# Run the agent
# -------------------------------
if __name__ == "__main__":
    user_query = input("Enter your literature review topic: ")
    result = review_papers(user_query)



