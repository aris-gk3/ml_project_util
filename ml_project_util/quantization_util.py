import json
from .path import path_definition

def save_range(min_in, max_in, model_name, layer_name):
    BASE_PATH, PATH_DATASET, PATH_RAWDATA, PATH_JOINEDDATA, PATH_SAVEDMODELS = path_definition()
    filepath = f'{BASE_PATH}/Docs_Reports/Quant/activation_range_{model_name}.json'
    try:
        with open(filepath, 'r') as file:
            layer_min_max = json.load(file)
    except:
        layer_min_max = {}

    layer_min_max[layer_name] = (min_in, max_in)

    with open(filepath, 'w') as file:
        json.dump(layer_min_max, file, indent=4)
