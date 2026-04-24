import asyncio
import json
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from dsg_core import DynamicSceneGraph
from llm_client import LLMClient

app = FastAPI(title="FusionX Engine")

# Allow the React Frontend to connect to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our core proprietary algorithm (The FSM/DSG)
dsg_engine = DynamicSceneGraph("config.json")

# Initialize the Cognitive Layer
llm_client = LLMClient()

# This queue holds the text tokens that need to be streamed to the React UI
commentary_queue = asyncio.Queue()

async def process_frame_data(frame_data: dict):
    """
    Background task: Runs the DSG math. If the Delta Filter triggers, it hits the LLM.
    """
    nodes = frame_data.get("nodes", {})
    
    # 1. Pass raw coordinates into the Scene Graph
    triggers = dsg_engine.evaluate_frame(nodes)
    
    # 2. Check if the Delta Filter caught a spatial change
    if triggers:
        for payload in triggers:
            print(f"[ENGINE] Trigger Fired! Payload: {json.dumps(payload)}")
            
            # 3. Pass the Enriched JSON to the LLM
            commentary = await llm_client.generate_commentary(payload)
            print(f"[ENGINE] LLM Output: {commentary}")
            
            # 4. Push to the streaming queue for the UI
            await commentary_queue.put(commentary)

@app.post("/ingest")
async def ingest_vision_data(request: Request, background_tasks: BackgroundTasks):
    """
    Layer 1 (Perception) hits this endpoint 60 times a second.
    To ensure zero latency, we process the math in a background task.
    """
    frame_data = await request.json()
    background_tasks.add_task(process_frame_data, frame_data)
    return {"status": "received"}

@app.get("/stream")
async def sse_stream(request: Request):
    """
    Layer 4 (Streaming): The React UI connects here to listen for live commentary.
    Uses Server-Sent Events (SSE) to push text instantly without polling.
    """
    async def event_generator():
        while True:
            # If the client disconnects, stop sending events
            if await request.is_disconnected():
                break

            # Wait for the LLM to generate commentary
            commentary = await commentary_queue.get()
            
            # Yield the commentary chunk to the UI
            yield {
                "event": "commentary",
                "data": commentary
            }
            
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
