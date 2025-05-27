import json
import math
from tensorflow.keras.layers import Conv2D, Dense
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


def smallest_power_of_two_to_exceed(range, next_value):
    if range <= 0 or next_value <= 0:
        raise ValueError("Values must be greater than 0")
    
    options = []

    print(f'We have range: {range} & next value {next_value}')

    if next_value < range:
        ratio = range / next_value
        exponent = math.ceil(math.log2(ratio))
        factor = 2 ** (exponent-1)
        result = range / factor
        print(f'Result 1: {result}')
        if result>next_value:
            options.append(result)
    elif next_value > range:
        ratio = next_value / range
        exponent = math.ceil(math.log2(ratio))
        factor = 2 ** exponent
        result = range * factor
        print(f'Result 2: {result}')
        if result>next_value:
            options.append(result)
    
    if not options:
        print(f'Choose same ramge')
        return range

    return min(options)


def find_hw_range(model, input_range, range_dict_path, shift_range_path):
    input_min = input_range[0]
    input_max = input_range[1]
    # read json
    try:
        with open(range_dict_path, 'r') as file:
            layer_min_max = json.load(file)
    except:
        print('No float range dictionary found!')
    
    # compare and write dict
    layers_list = list(layer_min_max.keys())
    hw_range = {}
    hw_range[layers_list[0]] = {'max': input_max}
    hw_range[layers_list[0]]['min'] = input_min
    i = 0
    for layer in model.layers:
        # if isinstance(layer, Conv2D) or isinstance(layer, Dense):
        if i+1 < len(layers_list):
            tmp_max = \
                smallest_power_of_two_to_exceed(hw_range[layers_list[i]]['max'], layer_min_max[layers_list[i+1]]['max'])
            hw_range[layers_list[i+1]] = {'max': tmp_max}
            hw_range[layers_list[i+1]]['min'] = 0
            i = i + 1
    
    # write json
    hw_range_serializable = {
        layer: {
            "min": float(stats["min"]),
            "max": float(stats["max"])
        }
        for layer, stats in hw_range.items()
    }
    with open(shift_range_path, "w") as f:
        json.dump(hw_range_serializable, f, indent=4)