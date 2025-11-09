import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

arx_wrapper = ArxivAPIWrapper(top_k_results=10, doc_content_chars_max=2500)
arxiv = ArxivQueryRun(api_wrapper=arx_wrapper, description="Searching relevant research papers on arXiv.")

# # tavily = TavilySearchResults()
tavily = TavilySearch(max_results=3, search_depth="basic")

wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=150)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper, description="Searching relevant information on Wikipedia.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_tokens=None,
)

# arxiv_results = arxiv.invoke("Top papers on Quantum Computing")
# print("----------------------------------------------------------------------------------------------\n")
# print(arxiv_results)

# print("----------------------------------------------------------------------------------------------\n")
# wiki_results = wiki.invoke("Python programming language")
# print(wiki_results)

# print("----------------------------------------------------------------------------------------------\n")
# tavily_results = tavily.invoke({"query": "Latest advancements in artificial intelligence"})
# # tavily_results = tavily.invoke("Latest advancements in artificial intelligence")
# print(tavily_results)

tools = [arxiv, tavily, wiki]
llm_with_tools = llm.bind_tools(tools)

if __name__ == "__main__":  
    results = llm_with_tools.invoke("Explain the concept of reinforcement learning and provide recent research papers on knowledge graphs.")
    print(results)
