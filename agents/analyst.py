from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0
    )

def analyze_root_cause(anomaly_report: str, machine_type: str):
    llm = get_llm()
    
    messages = [
        SystemMessage(content="""You are ANALYST, an expert industrial 
failure analysis agent. Based on sensor anomaly reports, identify:
1. Most likely root cause (be specific)
2. Failure mode (e.g., bearing wear, seal leak, overheating)
3. Estimated time to failure if untreated
4. Risk level to production
Reason step by step like a senior engineer."""),
        HumanMessage(content=f"""
Machine Type: {machine_type}
Anomaly Report: {anomaly_report}

What is the root cause and how urgent is this?
""")
    ]
    
    response = llm.invoke(messages)
    return response.content