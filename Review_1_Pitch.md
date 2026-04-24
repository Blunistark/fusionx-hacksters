# Review 1: Idea Understanding & Architecture

**Purpose:** Validate approach direction  
**Hackathon Phase:** Day-1 (12:30 PM – 2:30 PM)  

---

## 1. Introduction to Problem Statement
*(Criteria: Understanding of Problem Statement - 5 Marks)*
*   **The Core Problem:** High latency in analyzing dense, high-resolution video streams limits the viability of live applications.
*   **The Context:** Live sports broadcasting (specifically Cricket), where events unfold in milliseconds.
*   **The Objective:** Optimize video understanding architectures for rapid streaming inference to achieve near-instantaneous live event analysis and automated commentary.

## 2. The Solution
*(Criteria: Innovation in Approach - 5 Marks)*
*   **The Innovation:** "Dimensional Reduction & Semantic FSM."
*   Instead of feeding massive video frames directly into heavy Video-LLMs (which causes massive latency), we decouple perception from cognition.
*   We use high-speed edge CV models to extract raw coordinates (physics), immediately compress that into kilobytes of JSON data, and pass it through a proprietary **Finite State Machine (FSM)**.
*   The FSM translates spatial coordinates into "Semantic Events" (e.g., `BALL_PITCHED`, `SHOT_PLAYED`), which are fed into a lightweight LLM for real-time narrative generation.

## 3. Progress till Review 1
*   Deep analysis of the problem statement and alignment with the cricket commentary use case.
*   Complete mapping of the system architecture and memory management framework.
*   Strategic scoping: Decided to build a **Fixed Camera MVP** to guarantee a working prototype within the 24-hour limit, while mapping out a **Multi-Camera Sensor Fusion plan** for enterprise scaling.
*   Defined the full list of 6-phase cricket micro-events our system needs to track.

## 4. System Architecture Design
*(Criteria: Solution Planning & Architecture Idea - 5 Marks)*
Our pipeline operates in 4 distinct layers:
1.  **Layer 1: Perception (Speed of Sight):** YOLO tracking players and the ball on a fixed 2D plane, streaming coordinate telemetry.
2.  **Layer 2: The Aggregator (The Algorithm):** A Finite State Machine that monitors the coordinate stream and triggers semantic events based on spatial intersections.
3.  **Layer 3: Context Memory (Speed of Thought):** A 3-tiered memory system (Match State, Current Over, Match Summary) to prevent the LLM from losing context or hallucinating.
4.  **Layer 4: Streaming Inference:** Real-time WebSockets/SSE streaming the generated commentary to the UI.
*   **Phase 2 Outlook (Enterprise Scale):** We have mapped out an enterprise scaling architecture using Homography and Sensor Fusion to ingest multiple cameras into a single unified 3D pitch map. 

## 5. Technology Stack Usage
*   **Perception:** YOLOv8 / MediaPipe Pose for high-fps object tracking.
*   **Message Broker:** Redis Streams / MQTT (In-memory, zero latency).
*   **Aggregator / Backend:** Python (FastAPI/Go) for the FSM logic.
*   **Cognition:** Low-latency LLM (Gemini 1.5 Flash / GPT-4o-mini).
*   **Frontend:** React / Vite with WebSockets for live commentary streaming.

## 6. Prototype Development Progress
*   Finalized system blueprint and data schemas.
*   Currently initializing the Python environments and setting up the YOLO inference loop on a sample test video.
*   Drafting the conditional logic for the FSM (e.g., mapping ball trajectory drops to pitch events).

## 7. Team Coordination
*(Criteria: Feasibility within 24 Hours - 5 Marks)*
To guarantee delivery in 24 hours, the pipeline allows us to work entirely in parallel:
*   **Member 1 (Perception):** Focuses entirely on getting YOLO to output `(x, y)` coordinates of the ball/players.
*   **Member 2 (FSM / Backend):** Builds the Python logic to read those coordinates and emit JSON events.
*   **Member 3 (LLM / Frontend):** Builds the React UI and the LLM prompt engineering, using mock JSON events while the CV pipeline is built.

## 8. Improvement over Initial Idea
*   **Initial Thought:** Feed raw frames to an LLM for commentary. **Pivot:** This would be too slow. Built the FSM middleware layer to compress data, solving the core "High Latency" constraint of the problem statement.
*   **Initial Thought:** Try to fuse 4 cameras. **Pivot:** To ensure a fully functional prototype for the judges, we pivoted to a high-fidelity **Fixed Camera MVP**, while packaging the multi-camera approach as our Phase 2 enterprise vision.

## 9. Conclusion till Review 2
*   **Current Status:** Architecture locked, roles distributed, and core algorithm defined. 
*   **Goal for Review 2:** Have the perception pipeline successfully tracking a video and the FSM successfully triggering at least 3 discrete semantic events (e.g., Bowler Release, Pitch, Hit) in the backend console.
