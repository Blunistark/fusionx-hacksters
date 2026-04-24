import json
import math

class DynamicSceneGraph:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.previous_edges = set()

    def calculate_distance(self, boxA, boxB):
        # Calculate center points
        cx_A = (boxA[0] + boxA[2]) / 2
        cy_A = (boxA[1] + boxA[3]) / 2
        cx_B = (boxB[0] + boxB[2]) / 2
        cy_B = (boxB[1] + boxB[3]) / 2
        return math.sqrt((cx_A - cx_B)**2 + (cy_A - cy_B)**2)

    def check_intersection(self, boxA, boxB):
        # AABB Intersection check
        return not (boxA[2] < boxB[0] or boxA[0] > boxB[2] or boxA[3] < boxB[1] or boxA[1] > boxB[3])

    def evaluate_frame(self, nodes):
        """
        Nodes expected format: { "Ball": {"box": [x1, y1, x2, y2], "velocity": 120}, "Bat": {"box": [...]}, ... }
        """
        current_edges = set()
        triggered_events = []

        for trigger in self.config["triggers_to_watch"]:
            node_A_type = trigger["node_A_type"]
            node_B_type = trigger.get("node_B_type")
            condition = trigger["condition"]

            if condition == "PRESENCE":
                if node_A_type in nodes:
                    edge_id = f"{node_A_type}_PRESENCE"
                    current_edges.add(edge_id)
                    
                    if edge_id not in self.previous_edges:
                        payload = {
                            "event": trigger["trigger_name"],
                            "primary_actor": { "type": node_A_type, **nodes[node_A_type] }
                        }
                        triggered_events.append(payload)

            elif node_A_type in nodes and node_B_type in nodes:
                boxA = nodes[node_A_type]["box"]
                boxB = nodes[node_B_type]["box"]

                if condition == "INTERSECTS":
                    if self.check_intersection(boxA, boxB):
                        edge_id = f"{node_A_type}_{condition}_{node_B_type}"
                        current_edges.add(edge_id)
                        
                        # THE DELTA FILTER: Only fire if this edge didn't exist in the previous frame
                        if edge_id not in self.previous_edges:
                            # Package the Enriched JSON Payload
                            payload = {
                                "event": trigger["trigger_name"],
                                "primary_actor": { "type": node_A_type, **nodes[node_A_type] },
                                "secondary_actor": { "type": node_B_type, **nodes[node_B_type] }
                            }
                            triggered_events.append(payload)

        # Update state
        self.previous_edges = current_edges
        return triggered_events

# Example Usage
if __name__ == "__main__":
    dsg = DynamicSceneGraph("config.json")
    
    # Mock frame 1: Ball and bat are far apart
    frame1_nodes = {
        "Ball": {"box": [10, 10, 20, 20], "velocity_kph": 140},
        "Bat": {"box": [100, 100, 120, 150], "bat_speed_kph": 80}
    }
    print("Frame 1 Triggers:", dsg.evaluate_frame(frame1_nodes)) # Should be empty
    
    # Mock frame 2: Ball and bat intersect
    frame2_nodes = {
        "Ball": {"box": [105, 110, 115, 120], "velocity_kph": 140},
        "Bat": {"box": [100, 100, 120, 150], "bat_speed_kph": 80}
    }
    print("Frame 2 Triggers:", dsg.evaluate_frame(frame2_nodes)) # Should fire SHOT_PLAYED
    
    # Mock frame 3: Ball and bat still intersecting (Follow-through)
    print("Frame 3 Triggers:", dsg.evaluate_frame(frame2_nodes)) # Delta filter blocks it! Should be empty
