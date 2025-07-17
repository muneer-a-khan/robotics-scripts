import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict

def create_snap_circuit_board_background(ax, width, height):
    """Create a snap circuit board background with hexagonal grid pattern"""
    # Set background color to light gray/white
    ax.set_facecolor('#F8F8F8')
    
    # Board border
    board = patches.Rectangle((0, 0), width, height, 
                            linewidth=3, edgecolor='#CCCCCC', 
                            facecolor='#F0F0F0', alpha=0.8)
    ax.add_patch(board)
    
    # Create hexagonal grid pattern
    hex_size = min(width, height) * 0.03  # Adjust size based on image
    hex_spacing_x = hex_size * 1.5
    hex_spacing_y = hex_size * np.sqrt(3)
    
    # Calculate grid dimensions
    rows = int(height / hex_spacing_y) + 2
    cols = int(width / hex_spacing_x) + 2
    
    for row in range(rows):
        for col in range(cols):
            # Offset every other row for hexagonal pattern
            x_offset = (hex_spacing_x / 2) if row % 2 == 1 else 0
            x = col * hex_spacing_x + x_offset
            y = row * hex_spacing_y
            
            if 0 <= x <= width and 0 <= y <= height:
                # Create hexagon
                angles = np.linspace(0, 2*np.pi, 7)
                hex_x = x + hex_size * 0.4 * np.cos(angles)
                hex_y = y + hex_size * 0.4 * np.sin(angles)
                
                hexagon = patches.Polygon(list(zip(hex_x, hex_y)), 
                                        closed=True, linewidth=0.5, 
                                        edgecolor='#DDDDDD', 
                                        facecolor='none', alpha=0.6)
                ax.add_patch(hexagon)
                
                # Add small connection points
                circle = patches.Circle((x, y), hex_size * 0.1, 
                                      facecolor='#BBBBBB', 
                                      edgecolor='#999999', 
                                      linewidth=0.5, alpha=0.7)
                ax.add_patch(circle)

# Load the JSON
with open("output/data/graph-output-example.json") as f:
    data = json.load(f)

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

# Filter components with confidence > 75%
high_confidence_components = []
for component in data["connection_graph"]["components"]:
    if component["confidence"] > 0.75:
        high_confidence_components.append(component)

print(f"Showing {len(high_confidence_components)} components with >75% confidence")
for comp in high_confidence_components:
    print(f"  {comp['component_type']}: {comp['confidence']:.3f}")

G = nx.Graph()

# Mapping from point to components that use it
point_to_components = defaultdict(list)

# Get image dimensions from bbox coordinates
all_coords = []
for component in high_confidence_components:
    bbox = component["bbox"]
    all_coords.extend([bbox[0], bbox[1], bbox[2], bbox[3]])

min_x, max_x = min(all_coords[::2]), max(all_coords[::2])
min_y, max_y = min(all_coords[1::2]), max(all_coords[1::2])
img_width = max_x - min_x
img_height = max_y - min_y

print(f"Image dimensions: {img_width:.0f} x {img_height:.0f}")

# Step 1: Add nodes and track connection points
component_info = {}
positions = {}

for component in high_confidence_components:
    comp_id = component["id"]
    label = component["component_type"]
    confidence = component["confidence"]
    bbox = component["bbox"]
    
    # Calculate center position from bbox
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Convert to relative coordinates (0-1 range)
    rel_x = (center_x - min_x) / img_width
    rel_y = 1 - ((center_y - min_y) / img_height)  # Flip Y to match matplotlib
    
    positions[comp_id] = (rel_x, rel_y)
    
    num_connections = len(component["connection_points"])
    
    # Store component info
    component_info[comp_id] = {
        'type': label,
        'confidence': confidence,
        'connections': num_connections,
        'bbox': bbox
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
                if components[i] in G.nodes() and components[j] in G.nodes():
                    G.add_edge(components[i], components[j])

# Step 3: Create the visualization
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Create snap circuit board background
create_snap_circuit_board_background(ax, 1.0, 1.0)

# Get node attributes
node_colors = [G.nodes[node]['color'] for node in G.nodes()]
node_labels = {}
for node in G.nodes():
    info = component_info[node]
    # Create labels with component info
    node_labels[node] = f"{info['type']}\nconf: {info['confidence']:.2f}"

# Draw edges first (so they appear behind nodes)
for edge in G.edges():
    if edge[0] in positions and edge[1] in positions:
        x_coords = [positions[edge[0]][0], positions[edge[1]][0]]
        y_coords = [positions[edge[0]][1], positions[edge[1]][1]]
        ax.plot(x_coords, y_coords, 
               color='#666666', linewidth=3, alpha=0.8, zorder=2)

# Draw nodes at their actual positions
for i, node in enumerate(G.nodes()):
    if node in positions:
        x, y = positions[node]
        color = G.nodes[node]['color']
        
        # Draw component circle
        circle = patches.Circle((x, y), 0.04, 
                              facecolor=color, 
                              edgecolor='black',
                              linewidth=2, alpha=0.9, zorder=3)
        ax.add_patch(circle)
        
        # Add label
        info = component_info[node]
        ax.text(x, y-0.08, node_labels[node], 
               ha='center', va='top', fontsize=10, 
               fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='white', alpha=0.8),
               zorder=4)

# Create a legend for component types
legend_elements = []
unique_types = set(component_info[comp]['type'] for comp in component_info)
for comp_type in sorted(unique_types):
    color = component_colors.get(comp_type, default_color)
    legend_elements.append(plt.scatter([], [], c=color, s=200, label=comp_type))

ax.legend(handles=legend_elements, 
         title="Component Types (>75% confidence)",
         bbox_to_anchor=(1.02, 1), 
         loc='upper left',
         fontsize=12)

ax.set_title("Snap Circuit Board - Component Layout\n(Positioned based on actual detection locations)", 
            fontsize=18, fontweight='bold', pad=20)

# Add statistics
total_components = len(G.nodes())
total_connections = len(G.edges())
ax.text(0.5, -0.05, f"High Confidence Components: {total_components} | Connections: {total_connections}", 
       ha='center', va='top', transform=ax.transAxes,
       fontsize=14, style='italic')

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.15, 1.1)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()

# Save the graph
plt.savefig("output/snap_circuit_board_layout.png", dpi=300, bbox_inches='tight')
plt.show()

# Print detailed component statistics
print(f"\n=== High Confidence Component Details ===")
for comp_id, info in component_info.items():
    print(f"{comp_id}: {info['type']} (confidence: {info['confidence']:.3f}, connections: {info['connections']})")
    print(f"  Position: {positions[comp_id][0]:.3f}, {positions[comp_id][1]:.3f}")
    print(f"  Bbox: {info['bbox']}")
    print() 