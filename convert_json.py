import json
import sys

def convert_to_ocpq(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    valid_nodes = []
    for node in data.get('nodes', []):
        if node.get('type') == 'eventType' and 'data' in node and 'box' in node['data']:
            valid_nodes.append(node)
            
    id_to_index = {node['id']: idx for idx, node in enumerate(valid_nodes)}
    edges = data.get('edges', [])
    
    children_map = {node['id']: [] for node in valid_nodes}
    for edge in edges:
        source_id = edge.get('source')
        target_id = edge.get('target')
        if source_id in children_map and target_id in id_to_index:
            children_map[source_id].append(id_to_index[target_id])
            
    ocpq_nodes = []
    for node in valid_nodes:
        box_data = node['data']['box']
        children_indices = children_map[node['id']]
        ocpq_nodes.append({"Box": [box_data, children_indices]})
        
    edge_names = {}
    for edge in edges:
        source_id = edge.get('source')
        target_id = edge.get('target')
        name = edge.get('data', {}).get('name')
        
        if source_id in id_to_index and target_id in id_to_index and name:
            source_idx = id_to_index[source_id]
            target_idx = id_to_index[target_id]
            edge_names[f"{source_idx},{target_idx}"] = name

    # edge_names in Rust struct uses `#[serde_as(as = "Vec<(_, _)>")]`
    # meaning it should be serialized as an array of tuples (arrays in JSON)
    formatted_edge_names = [[[int(k.split(',')[0]), int(k.split(',')[1])], v] for k, v in edge_names.items()]
            
    result = {
        "nodes": ocpq_nodes,
        "edgeNames": formatted_edge_names
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        
    print(f"Successfully converted ReactFlow JSON to flattened OCPQ syntax tree -> {output_file}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        pass
        # print("Usage: python convert_json.py <input.json> <output.json>")
        # sys.exit(1)
        
    # convert_to_ocpq(sys.argv[1], sys.argv[2])
    convert_to_ocpq(sys.argv[1] if len(sys.argv) > 1 else 'raw.json', sys.argv[2] if len(sys.argv) > 2 else 'test.json')
