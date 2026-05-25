import json
import os 
import itertools

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def reset_config():
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'params_golden_1.json')
    config=read_json_file(relative_path)
    relative_path = os.path.join(current_dir, 'params.json')
    write_json_file(config, relative_path)

def reset_config_mid(dse_count):
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'params_golden_1.json')
    config=read_json_file(relative_path)
    ranges = {key: [value * (2 ** i) for i in range(dse_count)] for key, value in config.items()}
    combinations = list(itertools.product(*ranges.values()))
    all_configs = [dict(zip(ranges.keys(), combo)) for combo in combinations]
    #import pdb;pdb.set_trace()
    relative_path = os.path.join(current_dir, 'params.json')
    write_json_file(all_configs[len(all_configs)//2], relative_path)


