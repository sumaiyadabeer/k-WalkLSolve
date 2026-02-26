import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import os

def load_csr_from_txt(file_path):
    print(f"Parsing graph file: {file_path}")
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Clean up lines
        lines = [line.strip() for line in lines if line.strip()]

        # 1. Read V (Number of Vertices)
        V, E = map(int, lines[0].split())
        
        # Helper function to find and parse data lines
        def parse_data_line(header, data_type=int):
            try:
                # Find the index of the header line (e.g., "row_ptr")
                header_index = lines.index(header)
                # The next line is the data
                data_line = lines[header_index + 1]
                # Split by space and convert to the specified data type
                return np.array([data_type(x) for x in data_line.split()])
            except ValueError:
                print(f"Error: Could not find or parse header '{header}'.")
                return None
            except IndexError:
                print(f"Error: Data line missing after header '{header}'.")
                return None

        # 2. Parse row_ptr and col_off
        # row_ptr = parse_data_line("row_ptr", data_type=int)
        data_line = lines[1]
        row_ptr = np.array([int(x) for x in data_line.split()])
        # col_off = parse_data_line("col_off", data_type=int)
        data_line = lines[2]
        col_off = np.array([int(x) for x in data_line.split()])
        
        if row_ptr is None or col_off is None:
            raise ValueError("Could not parse row_ptr or col_off.")

        # 3. Parse values (optional)
        values = parse_data_line("values", data_type=float)
        if values is None:
            print("No 'values' line found. Assuming unweighted graph (all edges = 1).")
            # Create an array of 1s, with the same length as col_off
            values = np.ones(len(col_off), dtype=int)
        
        # 4. Create the CSR matrix
        # csr_matrix((data, indices, indptr), shape=(M, N))
        csr_mat = csr_matrix((values, col_off, row_ptr), shape=(V, V))
        
        print("Successfully parsed file and created CSR matrix.")
        return csr_mat

    except FileNotFoundError:
        print(f"Error: Graph file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None


def calculate_graph_diameter_from_csr(csr_mat):
    # 1. Convert the CSR matrix into a NetworkX Graph object.
    #    This interprets the matrix as an adjacency matrix.
    try:
        # For networkx 3.x and later
        G = nx.from_scipy_sparse_array(csr_mat)
    except AttributeError:
        # For older networkx versions
        G = nx.from_scipy_sparse_matrix(csr_mat)

    print(f"Graph loaded into NetworkX.")
    print(f"  - Number of nodes: {G.number_of_nodes()}")
    print(f"  - Number of edges: {G.number_of_edges()}")

    # 2. Check if the graph is connected.
    #    The diameter is only defined for a connected graph.
    #    If the graph is not connected, the diameter is infinite.
    if nx.is_connected(G):
        print("Graph is connected. Calculating diameter...")
        # 3. Calculate the diameter.
        #    This finds the "longest shortest path" between any two nodes.
        diameter = nx.diameter(G)
        print(f"Diameter calculated: {diameter}")
        return diameter
    else:
        print("Graph is not connected. The diameter is infinite.")
        
        # Optional: If you want to find the diameter of the largest component
        # Find all connected components and sort them by size
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        largest_component = G.subgraph(components[0])
        print(f"Calculating diameter of the largest connected component ({largest_component.number_of_nodes()} nodes)...")
        
        diameter_largest_component = nx.diameter(largest_component)
        print(f"Diameter of largest component: {diameter_largest_component}")
        
        return float('inf') # Return infinite for the whole graph

# --- Example Usage ---
if __name__ == "__main__":
    
    # --- 1. Create a Dummy 'graph.txt' file in your format ---
    # This creates the same "line" graph: 0-1-2-3-4 (Diameter = 4)
    graph_file_path = '/data1/sumaiya/LsolveVarient/LsolveMR_EAAI/MP/dataset_RHS/wikioc/oc_csr'
    
#     # Let's create the graph 0-1-2-3-4 correctly.
#     # Adjacency:
#     # 0: [1]
#     # 1: [0, 2]
#     # 2: [1, 3]
#     # 3: [2, 4]
#     # 4: [3]
    
#     # CSR representation:
#     V = 5
#     E = 8 # 4 edges, 2 directions
#     row_ptr_data = "0 1 3 5 7 8" 
#     # Node 0: col_off[0:1] = [1]
#     # Node 1: col_off[1:3] = [0, 2]
#     # Node 2: col_off[3:5] = [1, 3]
#     # Node 3: col_off[5:7] = [2, 4]
#     # Node 4: col_off[7:8] = [3]
#     col_off_data = "1 0 2 1 3 2 4 3"
#     values_data  = "1 1 1 1 1 1 1 1" # Unweighted

#     dummy_file_content = f"""
# {V} {E}
# row_ptr
# {row_ptr_data}
# col_off
# {col_off_data}
# values
# {values_data}
# """
#     try:
#         with open(graph_file_path, 'w') as f:
#             f.write(dummy_file_content)
#         print(f"Created dummy file: '{graph_file_path}'")
#     except Exception as e:
#         print(f"Error creating dummy file: {e}")

    # --- 2. Load the CSR matrix from the text file ---
    my_csr_matrix = load_csr_from_txt(graph_file_path)

    # --- 3. Calculate the diameter ---
    if my_csr_matrix is not None:
        print("\n--- Calculating Diameter ---")
        diameter = calculate_graph_diameter_from_csr(my_csr_matrix)
        print(f"\nFinal Diameter: {diameter}")