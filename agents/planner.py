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

def create_maintenance_plan(root_cause_report: str, machine_type: str):
    llm = get_llm()
    
    messages = [
        SystemMessage(content="""You are PLANNER, an expert maintenance 
planning agent. Based on root cause analysis, create a detailed plan:
1. Immediate actions (next 1 hour)
2. Short-term fix (next 24 hours) 
3. Parts needed (list specific parts)
4. Estimated downtime
5. Cost estimate (USD)
Be specific and actionable."""),
        HumanMessage(content=f"""
Machine Type: {machine_type}
Root Cause Report: {root_cause_report}

Create a detailed maintenance action plan.
""")
    ]
    
    response = llm.invoke(messages)
    return response.content