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

def analyze_sensor_data(sensor_df):
    llm = get_llm()
    
    stats = sensor_df.describe().round(2).to_string()
    recent = sensor_df.tail(10).to_string()
    
    messages = [
        SystemMessage(content="""You are SENTINEL, an industrial sensor anomaly 
detection agent. Analyze sensor data and identify:
1. Anomalies (values outside normal range)
2. Trend warnings (gradual degradation)  
3. Urgency level: CRITICAL / WARNING / NORMAL
Be concise and specific."""),
        HumanMessage(content=f"""
Sensor Statistics:
{stats}

Recent Readings (last 10):
{recent}

Identify anomalies and severity.
""")
    ]
    
    response = llm.invoke(messages)
    return response.content