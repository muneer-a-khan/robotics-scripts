#!/usr/bin/env python3
import json
from pathlib import Path

# Find the latest detection file
output_dir = Path("output/data")
detection_files = list(output_dir.glob("detection_*.json"))
latest_file = max(detection_files, key=lambda x: x.stat().st_mtime)

print(f"📊 ANALYSIS OF LATEST DETECTION")
print(f"📁 File: {latest_file.name}")
print("="*50)

# Load and analyze
with open(latest_file) as f:
    data = json.load(f)

components = data['connection_graph']['components']
edges = data['connection_graph']['edges']
state = data['connection_graph']['state']

print(f"🔧 COMPONENTS DETECTED: {len(components)}")
for i, comp in enumerate(components, 1):
    label = comp['label']
    conf = comp['confidence']
    print(f"  {i}. {label} (confidence: {conf:.3f})")

print(f"\n🔗 CONNECTIONS FOUND: {len(edges)}")
for i, edge in enumerate(edges, 1):
    comp1 = edge['component_1']
    comp2 = edge['component_2']
    print(f"  {i}. {comp1} ↔ {comp2}")

print(f"\n⚡ CIRCUIT STATE:")
print(f"  • Circuit closed: {state['is_circuit_closed']}")
print(f"  • Power on: {state['power_on']}")
print(f"  • Active components: {len(state['active_components'])}")

print(f"\n📈 PERFORMANCE:")
processing_time = data['processing_time']
print(f"  • Processing time: {processing_time:.3f}s")
print(f"  • Detection success: {'✅' if len(components) > 0 else '❌'}") 