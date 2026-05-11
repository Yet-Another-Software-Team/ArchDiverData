import os
import torch
import pandas as pd
import argparse

def is_smell_in_path(path, smell_comp, project_name=""):
    """
    Strictly checks if a smell component name matches a file path.
    Handles exact folder matches, namespaces, and Java's default package.
    """
    p = path.replace("\\", "/").lower()
    p_parts = p.split("/")
    s = str(smell_comp).lower()
    
    # Handle Java's (default package) by mapping it to the root src or project folder
    if s == "(default package)":
        if p_parts[-1] in ["src", "java", "main"] or p_parts[-1] == project_name.lower():
            return True

    # 1. Exact Literal Folder Name Match (Handles C# folders with dots like "ConsoleShopper.UI")
    if p_parts[-1] == s:
        return True
        
    # 2. Namespace / Multi-folder Match (e.g., "com.jica.chap09")
    if "." in s:
        s_slash = s.replace(".", "/")
        
        # Exact namespace match (e.g. ".../com/jica/chap09")
        if p.endswith(s_slash):
            return True
            
        # Fallback for Namespaces where the physical folders omit the root
        parts = s.split(".")
        if len(parts) > 1:
            # Check the last two parts of the namespace (e.g., "jica/chap09")
            last_two = f"{parts[-2]}/{parts[-1]}"
            if p.endswith(last_two):
                return True
                
            # Check just the very last part as a final fallback
            last_part = parts[-1]
            if p_parts[-1] == last_part:
                return True
                
    return False

def add_labels_to_graphs(csv_path, input_graph_dir, output_graph_dir):
    print(f"Loading labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    smells_by_project = df.groupby('name')['component'].apply(lambda x: list(x.dropna())).to_dict()

    if not os.path.exists(output_graph_dir):
        os.makedirs(output_graph_dir)

    x_counter = 1
    
    print(f"Processing graphs in {input_graph_dir}...\n")
    
    for filename in os.listdir(input_graph_dir):
        if not filename.endswith(".pt") or not filename.startswith("code_graph_"):
            continue
            
        project_name = filename.replace("code_graph_", "").replace(".pt", "")
        filepath = os.path.join(input_graph_dir, filename)
        
        loaded_obj = torch.load(filepath, weights_only=False)
        
        if isinstance(loaded_obj, dict) and 'graph_data' in loaded_obj:
            graph_data = loaded_obj['graph_data']
            component_mapping = loaded_obj.get('component_mapping', {})
        else:
            continue
            
        num_components = graph_data['Component'].num_nodes
        y_labels = torch.zeros(num_components, dtype=torch.long)
        
        smelly_components = smells_by_project.get(project_name, [])
        unmatched_smells = set(smelly_components)
        
        if smelly_components:
            # Sort paths by length so we prefer root-level / closer-to-root matches
            # over deeply nested subfolders with the same name.
            sorted_nodes = sorted(component_mapping.items(), key=lambda item: len(item[1]))
            
            for smell_comp in smelly_components:
                for node_id, path in sorted_nodes:
                    if is_smell_in_path(path, smell_comp, project_name):
                        y_labels[node_id] = 1 # 1 = has architectural smell
                        if smell_comp in unmatched_smells:
                            unmatched_smells.remove(smell_comp)
                        
                        # CRUCIAL FIX: Stop searching once we've found the best matching 
                        # folder for this specific smell. This completely prevents over-counting!
                        break 
                    
        graph_data['Component'].y = y_labels
        
        out_filename = f"graph_{x_counter}.pt"
        out_filepath = os.path.join(output_graph_dir, out_filename)
        
        torch.save({
            'graph_data': graph_data,
            'class_mapping': loaded_obj.get('class_mapping', {}),
            'component_mapping': component_mapping
        }, out_filepath)
        
        smells_found = int(y_labels.sum())
        print(f"[{out_filename}] Processed '{project_name}' -> {smells_found} smell(s) labeled.")
        
        if unmatched_smells:
            print(f"   --> WARNING: Could not find matching folders for: {unmatched_smells}")
        
        x_counter += 1

    print(f"\nSuccess! Labeled {x_counter - 1} graphs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='graph_labeler',
                    description='Maps architectural smell labels from a CSV to heterogeneous graph components.')

    parser.add_argument('-r','--root', help='Root directory containing the raw code graphs')
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
    else:
        output_dir = "labeled_graphs"

    add_labels_to_graphs(csv_file, root_dir, output_dir)