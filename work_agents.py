from langgraph.graph import StateGraph
from tools import llm,llm_with_tools
from dotenv import load_dotenv
import os
from typing import Literal , Annotated

from typing import TypedDict
from pydantic import BaseModel, Field 
import operator

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

class RouterAgentSchema(BaseModel):

    trigger_agent : Literal["ideation_agent", "literature_review_agent","conference_agent"] = Field(description="The agent to trigger based on the user's query.")
    # trigger_agent : Annotated[str, Field(description="The agent to trigger based on the user's query."),operator.update]

Router_Agent = llm.with_structured_output(RouterAgentSchema)

result = Router_Agent.invoke("I want to understand about knowledge graphs in short. which agent should I use?")
print(result)



