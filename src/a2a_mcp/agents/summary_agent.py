
import asyncio
from fastapi import FastAPI, Request
from a2a_mcp.common.utils import A2AClient
from a2a_mcp.common.prompts import SUMMARY_COT_INSTRUCTIONS
from google import genai

app = FastAPI()
client = A2AClient()

@app.get("/.well-known/agent.json")
async def get_agent_card():
    # In a real app, this would be loaded from a file
    return {"name": "SummaryAgent"}

@app.post("/run")
async def handle_request(request: Request):
    """
    Handles the main user request, orchestrates the planning and execution,
    and returns a final summary.
    """
    data = await request.json()
    user_query = data.get("user_query")
    results = data.get("results")

    client = genai.Client()
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=SUMMARY_COT_INSTRUCTIONS.replace(
            '{travel_data}', str(results)
        ),
        config={'temperature': 0.0},
    )
    summary = response.text

    return {"text": summary}
