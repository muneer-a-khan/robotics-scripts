import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load the JSON
with open("output/data/graph-output-example.json") as f:
    data = json.load(f)

G = nx.Graph()

# Define colors for different component types
component_colors = {
    'wire': '#FFD700',           # Gold
    'resistor': '#FF6B6B',       # Red
    'switch': '#4ECDC4',         # Teal
    'battery_holder': '#45B7D1', # Blue
    'button': '#96CEB4',         # Green
    'music_circuit': '#FFEAA7',  # Light Yellow
    'led': '#FD79A8',           # Pink
    'motor': '#A29BFE',         # Purple
    'lamp': '#FDCB6E',          # Orange
    'buzzer': '#6C5CE7',        # Violet
    'alarm': '#E17055',         # Dark Orange
    'photoresistor': '#74B9FF', # Light Blue
    'speaker': '#E84393'        # Magenta
}

# Default color for unknown components
default_color = '#95A5A6'  # Gray

# Mapping from point to components that use it
point_to_components = defaultdict(list)

# Step 1: Add nodes and track connection points
component_info = {}
for component in data["connection_graph"]["components"]:
    comp_id = component["id"]
    label = component["component_type"]
    confidence = component["confidence"]
    num_connections = len(component["connection_points"])
    
    # Store component info for tooltips
    component_info[comp_id] = {
        'type': label,
        'confidence': confidence,
        'connections': num_connections
    }
    
    G.add_node(comp_id, 
              label=label,
              confidence=confidence,
              connections=num_connections,
              color=component_colors.get(label, default_color))

    for pt in component["connection_points"]:
        pt = tuple(pt)
        point_to_components[pt].append(comp_id)

# Step 2: Add edges between components that share a point
for components in point_to_components.values():
    if len(components) > 1:
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                G.add_edge(components[i], components[j])

# Step 3: Draw the graph
plt.figure(figsize=(14, 10))

# Use a better layout algorithm
pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

# Get node attributes
node_colors = [G.nodes[node]['color'] for node in G.nodes()]
node_labels = {}
for node in G.nodes():
    info = component_info[node]
    # Create multi-line labels with component info
    node_labels[node] = f"{info['type']}\n{info['connections']} pts\n{info['confidence']:.2f}"

# Draw nodes with larger size and better styling
nx.draw_networkx_nodes(G, pos, 
                      node_color=node_colors, 
                      node_size=2000,
                      alpha=0.9,
                      linewidths=2,
                      edgecolors='black')

# Draw labels with better formatting
nx.draw_networkx_labels(G, pos, labels=node_labels, 
                       font_size=8, 
                       font_weight='bold',
                       font_color='black')

# Draw edges with better styling
nx.draw_networkx_edges(G, pos, 
                      edge_color='gray',
                      width=2,
                      alpha=0.7)

# Create a legend for component types
legend_elements = []
unique_types = set(component_info[comp]['type'] for comp in component_info)
for comp_type in sorted(unique_types):
    color = component_colors.get(comp_type, default_color)
    legend_elements.append(plt.scatter([], [], c=color, s=100, label=comp_type))

plt.legend(handles=legend_elements, 
          title="Component Types",
          bbox_to_anchor=(1.05, 1), 
          loc='upper left',
          fontsize=10)

plt.title("Circuit Component Connection Graph\n(Node size shows component, color shows type)", 
          fontsize=16, 
          fontweight='bold',
          pad=20)

# Add subtitle with statistics
total_components = len(G.nodes())
total_connections = len(G.edges())
plt.figtext(0.5, 0.02, f"Total Components: {total_components} | Total Connections: {total_connections}", 
           ha='center', fontsize=12, style='italic')

plt.axis('off')
plt.tight_layout()

# Save the graph
plt.savefig("output/circuit_network_graph.png", dpi=300, bbox_inches='tight')
plt.show()

# Print component statistics
print("\n=== Component Statistics ===")
type_counts = defaultdict(int)
for comp in component_info.values():
    type_counts[comp['type']] += 1

for comp_type, count in sorted(type_counts.items()):
    avg_confidence = np.mean([comp['confidence'] for comp in component_info.values() if comp['type'] == comp_type])
    avg_connections = np.mean([comp['connections'] for comp in component_info.values() if comp['type'] == comp_type])
    print(f"{comp_type}: {count} components, avg confidence: {avg_confidence:.3f}, avg connections: {avg_connections:.1f}") 