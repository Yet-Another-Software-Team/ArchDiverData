import argparse
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt

def visualize_hetero_code_graph(graph_data, class_mapping, component_mapping):
    G = nx.DiGraph()

    class_nodes = []
    clean_comp_nodes = []
    smelly_comp_nodes = []
    all_node_sizes = {} # Store sizes to help edge clipping

    has_labels = 'y' in graph_data['Component']
    y_labels = graph_data['Component'].y if has_labels else None

    # 1. Add Component Nodes
    for comp_id, path in component_mapping.items():
        node_name = f"Component_{comp_id}"
        label_name = os.path.basename(path)
        is_smelly = False
        if has_labels and comp_id < len(y_labels):
            if y_labels[comp_id].item() == 1:
                is_smelly = True

        G.add_node(node_name, label=label_name)
        if is_smelly:
            smelly_comp_nodes.append(node_name)
            all_node_sizes[node_name] = 1500
        else:
            clean_comp_nodes.append(node_name)
            all_node_sizes[node_name] = 1200

    # 2. Add Class Nodes
    for class_id, cls_name in class_mapping.items():
        node_name = f"Class_{class_id}"
        G.add_node(node_name, label=cls_name)
        class_nodes.append(node_name)
        all_node_sizes[node_name] = 600

    # 3. Add Edges
    print("--- Edge Extraction Debug ---")
    edge_count = 0
    
    # Using edge_items() is safer than iterating edge_types and checking keys
    for edge_type, storage in graph_data.edge_items():
        src_type, rel, dst_type = edge_type
        
        if hasattr(storage, 'edge_index') and storage.edge_index is not None:
            edge_index = storage.edge_index
            num_edges_in_type = edge_index.size(1)
            print(f"Found Edge Type {edge_type}: {num_edges_in_type} edges")
            
            for i in range(num_edges_in_type):
                # Ensure we are getting the integer ID
                u_id = edge_index[0, i].item()
                v_id = edge_index[1, i].item()
                
                # Correctly capitalize the node type for f-string construction
                src_node = f"{src_type.capitalize()}_{u_id}"
                dst_node = f"{dst_type.capitalize()}_{v_id}"
                
                G.add_edge(src_node, dst_node, type=rel)
                edge_count += 1
        else:
            print(f"Edge Type {edge_type} has no edge_index attribute.")

    print(f"Successfully added {edge_count} edges to NetworkX.")
    print("------------------------------")

    print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # --- PLOTTING ---
    plt.figure(figsize=(16, 12))
    # INCREASE k: this spreads nodes further apart so edges aren't hidden
    pos = nx.spring_layout(G, seed=42, k=1.5) 

    # Create a list of sizes for EVERY node in the graph order
    node_sizes = [all_node_sizes.get(n, 600) for n in G.nodes()]

    # DRAW EDGES FIRST
    nx.draw_networkx_edges(
        G, pos, 
        arrowstyle='-|>', 
        arrowsize=25, 
        edge_color='black', # Use high contrast
        width=1.5,
        node_size=node_sizes, # THIS IS THE FIX: Clips arrows to the node border
        alpha=0.4,
        connectionstyle="arc3,rad=0.1" # Curves the lines so they don't overlap
    )

    # DRAW NODES
    nx.draw_networkx_nodes(G, pos, nodelist=clean_comp_nodes, node_size=1200, node_color='#ffbf47', edgecolors='gray')
    nx.draw_networkx_nodes(G, pos, nodelist=smelly_comp_nodes, node_size=1500, node_color='#ff4d4d', edgecolors='darkred', linewidths=2)
    nx.draw_networkx_nodes(G, pos, nodelist=class_nodes, node_size=600, node_color='#66b3ff', edgecolors='gray')

    # DRAW LABELS
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')

    plt.title(f"Graph Visualization ({G.number_of_edges()} edges detected)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='graph_viewer',
                    description='Visualizes a heterogeneous graph from a .pt file.',
                    epilog='arguments: -f/--file (path to the graph file)')

    parser.add_argument('-f','--file', help='Path to the graph file to visualize')
    args = parser.parse_args()
    
    if not args.file:
        raise ValueError("Error: Please provide a graph file path using the -f or --file argument.")

    target_graph_file = args.file
    
    if os.path.exists(target_graph_file):
        print(f"Loading graph from {target_graph_file}...")
        
        saved_obj = torch.load(target_graph_file, weights_only=False)
        
        graph_data = saved_obj['graph_data']
        class_mapping = saved_obj['class_mapping']
        component_mapping = saved_obj['component_mapping']
        
        print("Graph loaded successfully!")
        visualize_hetero_code_graph(graph_data, class_mapping, component_mapping)
    else:
        print(f"Error: Could not find '{target_graph_file}'.")