import argparse
import os
import tomllib
import torch
from torch_geometric.data import HeteroData

def extract_class_features(toml_data):
    """
    Extracts a numeric feature vector based on the updated TOML format.
    """
    # 1. Class-level metrics
    class_lcom = toml_data.get("lcom", 0.0)
    attributes = toml_data.get("attribute", []) # Note: 'attribute' is singular in your TOML
    num_attributes = len(attributes)
    num_methods_declared = toml_data.get("num_methods", 0)

    # 2. Aggregate Method metrics
    methods = toml_data.get("methods", [])
    num_methods_actual = len(methods)

    total_method_lcom = 0.0
    total_params = 0

    for m in methods:
        total_method_lcom += m.get("lcom", 0.0)
        total_params += len(m.get("params", []))

    # Averages (safeguard against division by zero)
    avg_method_lcom = total_method_lcom / num_methods_actual if num_methods_actual > 0 else 0.0
    avg_params = total_params / num_methods_actual if num_methods_actual > 0 else 0.0

    # Feature vector representation (6 dimensions)
    return [
        float(class_lcom),
        float(num_attributes),
        float(num_methods_declared),
        float(num_methods_actual),
        float(avg_method_lcom),
        float(avg_params)
    ]

def build_hetero_code_graph(root_directory, output_file="code_graph.pt"):
    """
    Recursively parses TOML files matching the ChatTPIv1 schema, 
    builds a Heterogeneous PyTorch Geometric Data object, and saves it.
    """
    class_name_to_id = {}
    component_path_to_id = {}

    class_features_dict = {}
    component_features_dict = {}

    class_dependencies = {}
    
    edge_comp_contains_comp = ([], [])
    edge_comp_contains_class = ([], [])
    edge_class_contained_by_comp = ([], [])
    edge_class_imports_class = ([], [])

    print(f"--- Starting Pass 1: Parsing Directories and TOML files ---")
    
    # --- PASS 1: Recursively parse nodes, build containment edges ---
    for dirpath, dirnames, filenames in os.walk(root_directory):
        curr_dir = os.path.abspath(dirpath)
        
        # ADDED: Print statement to track which folder is currently being processed
        print(f"Processing directory: {curr_dir}")
        
        # 1. Register Component (Folder)
        if curr_dir not in component_path_to_id:
            component_path_to_id[curr_dir] = len(component_path_to_id)
        curr_comp_id = component_path_to_id[curr_dir]

        # Component Feature: Total number of files + folders inside
        valid_files = [f for f in filenames if f.endswith(".toml")]
        component_features_dict[curr_comp_id] = [float(len(dirnames) + len(valid_files))]

        # 2. Process subdirectories (Component contains Component)
        for dirname in dirnames:
            sub_dir = os.path.abspath(os.path.join(dirpath, dirname))
            if sub_dir not in component_path_to_id:
                component_path_to_id[sub_dir] = len(component_path_to_id)
            sub_comp_id = component_path_to_id[sub_dir]

            edge_comp_contains_comp[0].append(curr_comp_id)
            edge_comp_contains_comp[1].append(sub_comp_id)

        # 3. Process TOML files (Component contains Class)
        for filename in valid_files:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, "rb") as f:
                data = tomllib.load(f)

            # Fallback to filename if 'name' is missing, though your schema has it
            class_name = data.get("name", filename.replace(".toml", ""))
            if class_name not in class_name_to_id:
                class_name_to_id[class_name] = len(class_name_to_id)
            curr_class_id = class_name_to_id[class_name]

            # Extract features specific to the new TOML schema
            class_features_dict[curr_class_id] = extract_class_features(data)

            # Record Containment Edges (bi-directional)
            edge_comp_contains_class[0].append(curr_comp_id)
            edge_comp_contains_class[1].append(curr_class_id)
            edge_class_contained_by_comp[0].append(curr_class_id)
            edge_class_contained_by_comp[1].append(curr_comp_id)

            # Gather Imports directly from the 'imports' list
            # We convert to a set to remove duplicates
            class_dependencies[class_name] = set(data.get("imports", []))

    print(f"\n--- Starting Pass 2: Building Import Edges ---")
    
    # --- PASS 2: Build Class Imports Class Edges ---
    # 1. Create a "Simple Name" mapping to handle fully qualified imports
    # Example: Map "HomeController" -> id_0
    simple_name_to_id = {name.split('.')[-1]: id for name, id in class_name_to_id.items()}

    for src_class, deps in class_dependencies.items():
        src_id = class_name_to_id.get(src_class)
        if src_id is None: continue

        for dst_import_string in deps:
            # 2. Extract the actual Class Name from the end of the import
            # Example: "com.app.services.AuthService" -> "AuthService"
            dst_simple_name = dst_import_string.split('.')[-1]
            
            if dst_simple_name in simple_name_to_id:
                dst_id = simple_name_to_id[dst_simple_name]
                
                # Avoid self-loops
                if src_id != dst_id:
                    edge_class_imports_class[0].append(src_id)
                    edge_class_imports_class[1].append(dst_id)

    print(f"Finished building edges. Constructing PyTorch Geometric Data Object...")

    # --- Construct PyTorch Geometric HeteroData Object ---
    data = HeteroData()

    # 1. Build Node Feature Tensors
    num_classes = len(class_name_to_id)
    if num_classes > 0:
        c_x = [class_features_dict[i] for i in range(num_classes)]
        data['Class'].x = torch.tensor(c_x, dtype=torch.float)
    else:
        data['Class'].num_nodes = 0

    num_components = len(component_path_to_id)
    if num_components > 0:
        comp_x = [component_features_dict[i] for i in range(num_components)]
        data['Component'].x = torch.tensor(comp_x, dtype=torch.float)
    else:
        data['Component'].num_nodes = 0

    # 2. Build Edge Tensors
    def build_edge_tensor(edge_tuple):
        if edge_tuple[0]:
            return torch.tensor(edge_tuple, dtype=torch.long)
        return torch.empty((2, 0), dtype=torch.long)

    data['Component', 'contains', 'Component'].edge_index = build_edge_tensor(edge_comp_contains_comp)
    data['Component', 'contains', 'Class'].edge_index = build_edge_tensor(edge_comp_contains_class)
    data['Class', 'contained_by', 'Component'].edge_index = build_edge_tensor(edge_class_contained_by_comp)
    data['Class', 'imports', 'Class'].edge_index = build_edge_tensor(edge_class_imports_class)

    # --- Save Data ---
    torch.save({
        'graph_data': data,
        'class_mapping': {v: k for k, v in class_name_to_id.items()},
        'component_mapping': {v: k for k, v in component_path_to_id.items()}
    }, output_file)
    print(f"\nSUCCESS: Heterogeneous graph saved to {output_file}")
    print(f"Summary: {num_components} Components, {num_classes} Classes mapped.")

    return data, class_name_to_id, component_path_to_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='graph_creation',
                    description='Creates a heterogeneous graph from a directory of TOML files representing code structure.',
                    epilog='arguments: -r/--root (root directory containing the projects)')

    parser.add_argument('-r','--root', help='Root directory containing the projects')
    args = parser.parse_args()
    if args.root:
        root_dir = args.root
    else:
        raise ValueError("Error: Please provide a root directory using the -r or --root argument.")

    if not os.path.exists(root_dir):
        print(f"Error: The directory '{root_dir}' does not exist.")
    else:
        # Iterate through items directly inside the root_dir
        for item in os.listdir(root_dir):
            sub_dir_path = os.path.join(root_dir, item)
            
            # Check if the item is a folder (ignoring stray files in the root_dir)
            if os.path.isdir(sub_dir_path):
                print(f"\n" + "="*50)
                print(f"Processing Project: {item}")
                print("="*50)
                
                # Generate a unique output file name for this specific project
                # e.g., "code_graph_ProjectA.pt"
                output_filename = f"code_graph_{item}.pt"
                
                # Build and save the graph for this specific subdirectory
                build_hetero_code_graph(sub_dir_path, output_file=output_filename)