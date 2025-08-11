import asyncio
from fastapi import FastAPI, Request
from a2a_mcp.common.utils import A2AClient
from a2a_mcp.common.types import Plan, TripRequest

app = FastAPI()
client = A2AClient()

@app.get("/.well-known/agent.json")
async def get_agent_card():
    # In a real app, this would be loaded from a file
    return {"name": "OrchestratorAgent"}

@app.post("/run")
async def handle_request(request: Request):
    """
    Handles the main user request, orchestrates the planning and execution,
    and returns a final summary.
    """
    data = await request.json()
    user_query = data.get("message", {}).get("parts", [{}])[0].get("text", "")
    print(f"[Orchestrator] Received query: {user_query}")

    # 1. Delegate to Planner Agent to get a step-by-step plan
    print("[Orchestrator] Delegating to Planner Agent...")
    planner_payload = {"query": user_query}
    plan_response = await client.interact_with_agent("planner_agent", planner_payload)
    
    if "error" in plan_response:
        return {"summary": f"Failed to create a plan: {plan_response['error']}"}

    plan = Plan.model_validate(plan_response)
    print(f"[Orchestrator] Received plan: {plan.steps}")

    # 2. Execute the plan steps in parallel by calling worker agents
    worker_tasks = []
    for step in plan.steps:
        print(f"[Orchestrator] Delegating task '{step.task}' to {step.agent}")
        worker_payload = TripRequest(
            user_request=user_query,
            task=step.task,
            task_parameters=step.task_parameters,
        )
        worker_tasks.append(
            client.interact_with_agent(step.agent, worker_payload.model_dump())
        )

    print("[Orchestrator] Waiting for worker agents to complete...")
    worker_results = await asyncio.gather(*worker_tasks)
    print(f"[Orchestrator] All worker agents finished. Results: {worker_results}")

    # 3. Delegate to Summary Agent to create the final itinerary
    print("[Orchestrator] Delegating to Summary Agent...")
    summary_payload = {
        "user_query": user_query,
        "results": worker_results
    }
    summary_response = await client.interact_with_agent("summary_agent", summary_payload)
    
    final_summary = summary_response.get("text", "Could not generate summary.")
    print(f"[Orchestrator] Received final summary: {final_summary}")

    # 4. Return the final summary
    return {"summary": final_summary}