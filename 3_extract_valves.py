import pandas as pd
import numpy as np
import math
import os


VALVES_CSV = "output_image_1/valves_ocr_image_1_v1.csv" 
PIPES_CSV = "output_image_1/pipes_with_geometry_image_1.csv"  

OUT_CSV = "output_image_1/valve_pipe_connections_image_1.csv"
DEBUG_IMG = "output_image_1/debug_valve_pipe_connections_image_1.png"
# distance threshold (pixels)
# MAX_VALVE_PIPE_DIST = 25.0
MAX_VALVE_PIPE_DIST = 40.0

def point_to_line(px, py, x1, y1, x2, y2):
    """Distance from point to line segment"""
    num = abs((y2-y1)*px - (x2-x1)*py + x2*y1 - y2*x1)
    den = math.hypot(y2-y1, x2-x1) + 1e-6
    return num / den

# ================== MAIN ==================
print("Loading valves...")
valves = pd.read_csv(VALVES_CSV)
print(f"Found {len(valves)} valve components")

print("Loading pipes...")
pipes = pd.read_csv(PIPES_CSV)
print(f"Found {len(pipes)} pipe segments")

# Add pipe index
pipes["pipe_id"] = range(len(pipes))

connections = []
for _, valve in valves.iterrows():
    vx, vy = valve["center_x"], valve["center_y"]
    
    best_pipe_id = None
    best_dist = float('inf')
    
    for _, pipe in pipes.iterrows():
        dist = point_to_line(vx, vy, pipe["x1"], pipe["y1"], pipe["x2"], pipe["y2"])
        
        if dist < best_dist:
            best_dist = dist
            best_pipe_id = pipe["pipe_id"]
    
    if best_dist < MAX_VALVE_PIPE_DIST:
        best_pipe = pipes[pipes["pipe_id"] == best_pipe_id].iloc[0]
        
        connections.append({
            "valve_id": valve["valve_id"],
            "size": valve["size"],
            "service": valve["service"],
            "full_text": valve["full_text"],
            "valve_center_x": vx,
            "valve_center_y": vy,
            "pipe_id": best_pipe["pipe_id"],
            "pipe_tag": best_pipe.get("pipe_tag", f"P{best_pipe_id}"),
            "pipe_x1": best_pipe["x1"],
            "pipe_y1": best_pipe["y1"],
            "pipe_x2": best_pipe["x2"],
            "pipe_y2": best_pipe["y2"],
            "distance_to_pipe": best_dist,
        })

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame(connections).to_csv(OUT_CSV, index=False)

print(f"\n✅ {len(connections)} valves attached to pipes!")
print(f"→ {OUT_CSV}")
print("\nSample:")
print(pd.DataFrame(connections).head())
