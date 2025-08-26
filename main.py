import langgraph
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from tavily import TavilyClient
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

TAVILY_SEARCH_API = os.getenv("TAVILY_SEARCH_API")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tavily = TavilyClient(api_key=TAVILY_SEARCH_API)
llm = ChatOpenAI(model="gpt-4o-mini")

class AgentState(TypedDict):
    destination: str
    days: int
    attractions: List[Dict]
    urls: List[str]
    # hotels: List[Dict]
    # itinerary: Dict

@tool
def tavily_tool(query:str):
    """
    Search the web for attractions using Tavily API
    Args:
        query: search query used to search the web through tavily API
    """
    results = tavily.search(query=query, max_results=3)
    return [r["url"] for r in results["results"]]

def scrape_webpage(url: str) -> str:
    """
    Fetch webpage text and return cleaned content.
    
    Args:
        url: This is a url to the site that will be scraped using requests and beautifulsoup
    """
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        text = " ".join(soup.stripped_strings)
        return text[:5000]
    except Exception as e:
        return f"Error scraping {url}: {e}"

tools = [tavily_tool]
llm = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an AI that will use the provided information to find attraction sites at the destination of the user. Make use of the tavily_tool if needed for searching the web. Give me the search query for using this tool.
    """),
    ("human","""
        Destination: {destination}
        Days: {days}
        Attractions: {attractions}
        """
    )
])

def search_node(state: AgentState) -> AgentState:
    chain = prompt | llm
    response = chain.invoke({
        "destination": state['destination'],
        "days": state['days'],
        "attractions": state['attractions']
    })
    # print(response)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_response = tavily_tool(response.tool_calls[-1].get('args', {}).get('query', None))
        return {**state, "urls": tool_response}
    else:
        print(f"\nSearch node response: {response.content}\n")
        return {**state}

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an AI that will use the provided with some text to find attraction sites at the destination of the user. Make sure to use the the provided text and extract the names of the attraction sites. The user will provide the destination and text. The response should be a list of attraction sites for the given destination.
    """),
    ("human","""
        Destination: {destination}
        Text: {text}
        """
    )
])

class AttractionSites(BaseModel):
    attractions: List[str] = Field(..., description="List of must-see tourist attractions")

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(AttractionSites)

def extract_attractions_node(state: AgentState) -> AgentState:
    extracted_texts = []
    for url in state['urls']:
        web_content = scrape_webpage(url)
        extracted_texts.append(web_content)
    combined_text = "\n\n".join(extracted_texts)
    extraction_chain = extraction_prompt | structured_llm
    response: AttractionSites = extraction_chain.invoke({
        "destination": state["destination"],
        "text": combined_text
        })
    unique_attractions = list(set(state["attractions"] + response.attractions))
    return {**state, "attractions": unique_attractions}

graph = StateGraph(AgentState)

graph.add_node("attractions_node", search_node)
graph.set_entry_point("attractions_node")

graph.add_edge("attractions_node", END)

app = graph.compile()

inputs = {
    "destination": "Dubai",
    "days": 3,
    "attractions": []
}

response = app.invoke(inputs)

print(response["attractions"])