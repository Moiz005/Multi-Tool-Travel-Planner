import langgraph
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import requests

class AgentState(TypedDict):
    destination: str
    days: int
    attractions: List[Dict]
    hotels: List[Dict]
    itinerary: Dict
    