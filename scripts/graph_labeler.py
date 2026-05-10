import os
import torch
import pandas as pd
import argparse

def add_labels_to_graphs(csv_path, input_graph_dir, output_graph_dir):
    # 1. Load the training labels
    print(f"Loading labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Group the smelly component names by project 'name'
    # Result: {'sulavAryal-repo0': ['ConsoleShopper.UI'], 'self1': ['day3'], ...}
    smells_by_project = df.groupby('name')['component'].apply(lambda x: list(x.dropna())).to_dict()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_graph_dir):
        os.makedirs(output_graph_dir)

    x_counter = 1
    
    print(f"Processing graphs in {input_graph_dir}...\n")
    
    # 2. Iterate through all graphs
    for filename in os.listdir(input_graph_dir):
        if not filename.endswith(".pt") or not filename.startswith("code_graph_"):
            continue
            
        # Extract project name: "code_graph_ProjectName.pt" -> "ProjectName"
        project_name = filename.replace("code_graph_", "").replace(".pt", "")
        
        filepath = os.path.join(input_graph_dir, filename)
        
        # Load the graph (weights_only=False to bypass security check)
        loaded_obj = torch.load(filepath, weights_only=False)
        
        # Check if the mapping dictionary is present
        if isinstance(loaded_obj, dict) and 'graph_data' in loaded_obj:
            graph_data = loaded_obj['graph_data']
            component_mapping = loaded_obj.get('component_mapping', {})
        else:
            print(f"Warning: {filename} does not contain 'component_mapping'. Skipping...")
            continue
            
        # Initialize a tensor of zeros (0 = no smell) for all components
        num_components = graph_data['Component'].num_nodes
        y_labels = torch.zeros(num_components, dtype=torch.long)
        
        # Get the list of components that have smells for this specific project
        smelly_components = smells_by_project.get(project_name, [])
        
        # 3. Apply labels using the string mapping
        if smelly_components:
            for node_id, path in component_mapping.items():
                # Normalize path for safer matching
                path_normalized = path.replace("\\", "/")
                
                is_smelly = False
                for smell_comp in smelly_components:
                    # Depending on the language, 'com.example.screen' might be parsed as 
                    # a folder structure like 'com/example/screen'
                    smell_comp_slash = smell_comp.replace(".", "/")
                    
                    # Substring matching: Check if the folder path contains the smell name
                    if (smell_comp in path_normalized) or (smell_comp_slash in path_normalized):
                        is_smelly = True
                        break
                
                if is_smelly:
                    y_labels[node_id] = 1 # 1 = has architectural smell
                    
        # Attach the labels to the 'Component' node type natively in PyG
        graph_data['Component'].y = y_labels
        
        # 4. Save the newly labeled graph incrementally (graph_1.pt, graph_2.pt...)
        out_filename = f"graph_{x_counter}.pt"
        out_filepath = os.path.join(output_graph_dir, out_filename)
        
        # Save using the same dictionary format to preserve the mapping for the future
        torch.save({
            'graph_data': graph_data,
            'class_mapping': loaded_obj.get('class_mapping', {}),
            'component_mapping': component_mapping
        }, out_filepath)
        
        smells_found = int(y_labels.sum())
        print(f"[{out_filename}] Processed '{project_name}' -> {smells_found} smell(s) labeled.")
        
        x_counter += 1

    print(f"\nSuccess! Labeled {x_counter - 1} graphs and saved them to {output_graph_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='graph_creation',
                    description='Creates a heterogeneous graph from a directory of TOML files representing code structure.',
                    epilog='arguments: -r/--root (root directory containing the projects)')

    parser.add_argument('-r','--root', help='Root directory containing the projects')
    parser.add_argument('-i','--input', help='Name of the input CSV file')
    parser.add_argument('-o', '--output', help='Directory to save the labeled graphs')
    args = parser.parse_args()
    if args.root:
        root_dir = args.root
    else:
        raise ValueError("Error: Please provide a root directory using the -r or --root argument.")

    if args.input:
        csv_file = args.input
    else:
        csv_file = "training labels.csv"
    
    if args.output:
        output_dir = args.output

    add_labels_to_graphs(csv_file, root_dir, output_dir)