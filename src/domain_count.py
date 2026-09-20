import json
import os

def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from file: {file_path}\n{e}")
        
def count_domains(json_data):
    domain_count = {}
    
    for entry in json_data:
        domain = entry.get('ocel_path')
        if domain:
            domain_count[domain] = domain_count.get(domain, 0) + 1
            
    return domain_count

if __name__ == "__main__":
    test_set_path = "data/testset.json"
    
    try:
        test_set_data = load_json(test_set_path)
        domain_counts = count_domains(test_set_data)
        
        print("Domain Counts:")
        for domain, count in domain_counts.items():
            print(f"{domain}: {count}")
    except Exception as e:
        print(f"An error occurred: {e}")