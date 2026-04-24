# FusionX: Domain-Agnostic Real-Time Event Engine

## 1. Core Problem & Objective
*   **The Problem:** Traditional Video-LLMs suffer from massive latency when processing dense, high-resolution video streams (pixels), making them unusable for live applications.
*   **The Objective:** Achieve near-instantaneous live event analysis and commentary by compressing gigabytes of video into tiny, enriched text strings before feeding them to an AI.

## 2. The Innovation: Dimensional Reduction
We decouple **Perception** (Computer Vision) from **Cognition** (LLM). Instead of sending raw frames to an AI, we built a **Dynamic Scene Graph (DSG) Pipeline**:
1.  **Speed of Sight:** Lightweight edge models (YOLOv8) track object coordinates and speeds at 60 FPS.
2.  **The Engine:** A Python middleware calculates the geometric spatial relationships between these objects (e.g., `Is Touching`, `Is Near`).
3.  **The Delta Filter:** The engine ONLY sends data when a geometric relationship *changes* (a Graph Delta).
4.  **Enriched Payloads:** When a delta triggers, the engine compresses the event into a ~300-byte JSON string containing deep physics metadata (speeds, trajectories, colors).

## 3. The 4-Layer Architecture Pipeline
1.  **Layer 1: Perception Layer (Data Ingestion)**
    *   *Stack:* YOLOv8, MediaPipe, OpenCV.
    *   *Action:* Extracts raw bounding boxes (`[x,y,w,h]`) and pushes them to an in-memory broker.
2.  **Layer 2: Middleware & Data Pipeline (The Engine)**
    *   *Stack:* Python (FastAPI), Redis Streams.
    *   *Action:* The DSG algorithm evaluates spatial math and filters out noise. It packages the Enriched JSON Payload upon detecting an edge change.
3.  **Layer 3: Cognitive Layer (Trigger Response)**
    *   *Stack:* Gemini 1.5 Flash / GPT-4o-mini API.
    *   *Action:* Ingests the tiny JSON payload alongside the static match/domain context to generate intelligent narrative text.
4.  **Layer 4: Frontend & Streaming (UI)**
    *   *Stack:* React.js (Vite), Server-Sent Events (SSE).
    *   *Action:* Streams the LLM's text token-by-token directly to the dashboard, ensuring continuous live commentary.

## 4. Staggered Micro-Prompting
To make the AI sound like a continuous, live commentator (rather than a post-match summarizer), the DSG does not wait for an event to end. It fires **micro-prompts** multiple times a second:
*   *Trigger 1:* `[Event: Bowler Run-up]` -> AI speaks...
*   *Trigger 2:* `[Event: Ball Pitched]` -> AI speaks...
*   *Trigger 3:* `[Event: Shot Played]` -> AI speaks...

## 5. Domain-Agnostic Platform Vision
This architecture is a universal engine. 90% of the codebase remains identical regardless of the industry. To deploy to a new domain, we only swap:
1.  **The Vision Weights:** (e.g., tracking Cars instead of Cricket Balls).
2.  **The JSON Configuration File:** Defining new triggers (e.g., `Car TOUCHING Pedestrian` instead of `Ball TOUCHING Bat`).
3.  **The LLM System Persona:** (e.g., Traffic Security AI instead of Cricket Commentator).

## 6. Future Scope (Phase 3)
*   **Scene Graph Generation (SGG) Models:** Replacing the heuristic Python geometric math with a trained Relation Transformer (RelTR) to natively predict Scene Graphs directly from video frames, immunizing the system against camera angle distortion.
*   **Auto-Tuning Engine:** Using Bayesian Optimization or Reinforcement Learning to let the engine automatically tune its own spatial thresholds based on ground-truth human commentary.
