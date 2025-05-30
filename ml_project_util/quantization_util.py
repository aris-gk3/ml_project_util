import json
import math
import numpy as np
import os
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate # type: ignore
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from .path import path_definition
from .load_preprocess import load_preprocess


### Info & visualization for quantization

def subsample_imgdir(num_samples=100):
    train_dataset, val_dataset = load_preprocess()

    import os
    import random

    _, PATH_DATASET, _, _, _ = path_definition()
    img_dir1 = f'{PATH_DATASET}/DogConverted'
    img_dir2 = f'{PATH_DATASET}/CatConverted'
    num_samples = 40

    valid_extensions = ('.png', '.jpg', '.jpeg')

    # Initialize list
    all_files = []

    # Loop through both directories
    for img_dir in [img_dir1, img_dir2]:
        for f in os.listdir(img_dir):
            full_path = os.path.join(img_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(valid_extensions):
                all_files.append(full_path)
                
    random.seed(99)  # For reproducibility
    sampled_files = random.sample(all_files, num_samples)

    return sampled_files


def load_and_preprocess_image(file_path):
    img = tf.keras.utils.load_img(file_path, target_size=(224, 224, 3))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = preprocess_input(img_array)
    # img_array = img_array[..., ::-1]                                  # Reverses the last axis (channels), RGB -> BGR
    img_array = tf.expand_dims(img_array, axis=0)                       # Add batch dimension
    return img_array


def activation_statistics_md(model, sampled_files):
    # Initialize dict to store activations per layer
    layers_list = []
    for i in model.layers:
        if isinstance(i, Conv2D) or isinstance(i, Dense):
            layers_list.append(i.name)
    layer_activations = {layer_name: [] for layer_name in layers_list}

    # Process input files
    for file_path in sampled_files:
        x = load_and_preprocess_image(file_path)

        for layer in model.layers:
            x = layer(x)

            if layer.name in layers_list:
                act_np = x.numpy().flatten()
                layer_activations[layer.name].append(act_np)

    # Prepare data for tabulate
    table_data = []
    for layer_name, acts_list in layer_activations.items():
        all_acts = np.concatenate(acts_list)
        count = all_acts.size
        min_act = all_acts.min()
        max_act = all_acts.max()
        mean_act = all_acts.mean()
        std_act = all_acts.std()

        table_data.append([layer_name, count, min_act, max_act, mean_act, std_act])

    # Define headers
    headers = ["Layer", "Count", "Min", "Max", "Mean", "Std"]

    # Print table in markdown format
    print(tabulate(table_data, headers=headers, tablefmt="github", floatfmt=".4f"))


def wt_bias_statistics_md(model):
    weight_distributions = {}
    bias_distributions = {}

    for layer in model.layers:
        if hasattr(layer, "get_weights") and hasattr(layer, "set_weights"):
            weights = layer.get_weights()
            if weights:
                if len(weights) >= 1:
                    flat_weights = weights[0].flatten()
                    weight_distributions[layer.name] = flat_weights
                if len(weights) >= 2:
                    flat_biases = weights[1].flatten()
                    bias_distributions[layer.name] = flat_biases

    # Summarize weights
    weight_summary = []
    for layer, weights in weight_distributions.items():
        weights = np.array(weights)
        weight_summary.append([
            layer,
            len(weights),
            np.min(weights),
            np.max(weights),
            np.mean(weights),
            np.std(weights)
        ])

    # Summarize biases
    bias_summary = []
    for layer, biases in bias_distributions.items():
        biases = np.array(biases)
        bias_summary.append([
            layer,
            len(biases),
            np.min(biases),
            np.max(biases),
            np.mean(biases),
            np.std(biases)
        ])

    # Print as markdown tables
    print("### Weight Distributions per Layer")
    print(tabulate(weight_summary, headers=["Layer", "Count", "Min", "Max", "Mean", "Std"], tablefmt="github"))

    print("\n### Bias Distributions per Layer")
    print(tabulate(bias_summary, headers=["Layer", "Count", "Min", "Max", "Mean", "Std"], tablefmt="github"))

    return 0


def activation_dist_plots(sampled_files, model, model_name, mode='sv', filepath='0'):
    # s: save
    # v: verbose
    # sv: save & verbose

    tf.config.run_functions_eagerly(True)

    # First pass: collect min/max activation range per layer
    layer_min_max = {}
    layers_list = [layer.name for layer in model.layers if isinstance(layer, (Conv2D, Dense))]

    # Pass 1: Find min and max per layer
    for file_path in sampled_files:
        x = load_and_preprocess_image(file_path)

        for i, layer in enumerate(model.layers):
            x = layer(x)

            if layer.name in layers_list:
                act_min = tf.reduce_min(x).numpy()
                act_max = tf.reduce_max(x).numpy()

                if layer.name not in layer_min_max:
                    layer_min_max[layer.name] = {"min": act_min, "max": act_max}
                else:
                    layer_min_max[layer.name]['min'] = min(layer_min_max[layer.name]['min'], act_min)
                    layer_min_max[layer.name]['max'] = max(layer_min_max[layer.name]['max'], act_max)

    # Create bins for each layer
    num_bins = 128
    layer_histograms = {}
    for layer_name, stats in layer_min_max.items():
        bins = np.linspace(stats["min"], stats["max"], num_bins + 1)
        layer_histograms[layer_name] = {"bins": bins, "counts": np.zeros(num_bins, dtype=int)}

    # Second pass: fill histogram counts
    for file_path in sampled_files:
        x = load_and_preprocess_image(file_path)

        for i, layer in enumerate(model.layers):
            x = layer(x)

            if layer.name in layer_histograms:
                flat_activations = tf.reshape(x, [-1]).numpy()
                counts, _ = np.histogram(flat_activations, bins=layer_histograms[layer.name]["bins"])
                layer_histograms[layer.name]["counts"] += counts

    # Plot histograms
    for i, (layer_name, hist) in enumerate(layer_histograms.items()):
        bins = hist["bins"]
        counts = hist["counts"]

        # Compute bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        plt.figure(figsize=(8, 4))
        plt.bar(bin_centers, counts, width=(bins[1] - bins[0]), edgecolor='black')
        plt.yscale('log')  # set y-axis to log scale
        plt.title(f"Activation Histogram (log frequency) - {layer_name}")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency (log scale)")
        plt.grid(True, which='major', ls="--")
        plt.tight_layout()
        if mode=='s' or mode=='sv':
            if filepath=='0':
                BASE_PATH, _, _, _, _ = path_definition()
                parent_name = model_name[:3]
                short_name = model_name[:-10]
                plt.savefig(f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_layer{i:02d}_activation.png")
            else:
                plt.savefig(filepath)
        if mode=='v' or mode=='sv':
            plt.show()
        else:
            plt.close()


def wt_dist_plots(model, model_name, mode='sv', filepath='0'):
    # s: save
    # v: verbose
    # sv: save & verbose
    # Initialize dictionaries
    weight_distributions = {}
    bias_distributions = {}

    for layer in model.layers:
        if hasattr(layer, "get_weights") and hasattr(layer, "set_weights"):
            weights = layer.get_weights()
            if weights:
                if len(weights) >= 1:
                    flat_weights = weights[0].flatten()
                    weight_distributions[layer.name] = flat_weights
                if len(weights) >= 2:
                    flat_biases = weights[1].flatten()
                    bias_distributions[layer.name] = flat_biases

    BASE_PATH, PATH_DATASET, PATH_RAWDATA, PATH_JOINEDDATA, PATH_SAVEDMODELS = path_definition()
    parent_name = model_name[:3]
    short_name = model_name[:-10]
    # Plot weight distributions
    for i, (layer_name, weights) in enumerate(weight_distributions.items()):
        plt.figure(figsize=(6, 4))
        plt.hist(weights, bins=100, color='dodgerblue', alpha=0.7)
        plt.title(f"Weight Distribution: {layer_name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        if mode=='s' or mode=='sv':
            if filepath=='0':
                plt.savefig(f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_layer{i:02d}_wt.png")
            else:
                plt.savefig(filepath)
        if mode=='v' or mode=='sv':
            plt.show()
        plt.close()

    # Plot bias distributions
    for i, (layer_name, biases) in enumerate(bias_distributions.items()):
        plt.figure(figsize=(6, 4))
        plt.hist(biases, bins=100, color='tomato', alpha=0.7)
        plt.title(f"Bias Distribution: {layer_name}")
        plt.xlabel("Bias Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        if mode=='s' or mode=='sv':
            if filepath=='0':
                plt.savefig(f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_layer{i:02d}_bias.png")
            else:
                plt.savefig(filepath)
        if mode=='v' or mode=='sv':
            plt.show()
        plt.close()


def activation_violin_plot(sampled_files, model, model_name, mode='sv', filepath='0'):
    tf.config.run_functions_eagerly(True)

    layers_list = [layer.name for layer in model.layers if isinstance(layer, (Conv2D, Dense))]

    # Dict to accumulate activations per layer
    layer_activations = {layer_name: [] for layer_name in layers_list}

    # Collect activations (flattened) from all files per layer
    for file_path in sampled_files:
        x = load_and_preprocess_image(file_path)
        for layer in model.layers:
            x = layer(x)
            if layer.name in layers_list:
                flat_acts = tf.reshape(x, [-1]).numpy()
                layer_activations[layer.name].append(flat_acts)

    # Concatenate activations per layer into single arrays
    for layer_name in layer_activations:
        layer_activations[layer_name] = np.concatenate(layer_activations[layer_name])

    # Prepare data for seaborn violin plot
    import pandas as pd

    data = []
    for layer_name, acts in layer_activations.items():
        # Limit number of points if data is huge (optional)
        if len(acts) > 10000:
            acts = np.random.choice(acts, 10000, replace=False)
        for val in acts:
            data.append({"Layer": layer_name, "Activation": val})

    df = pd.DataFrame(data)

    # Plot violin plot
    plt.figure(figsize=(10, 8))
    sns.violinplot(y="Layer", x="Activation", data=df, scale='width', inner='quartile')
    plt.title("Activation Distributions per Layer")
    plt.xlabel("Activation Value")
    plt.ylabel("Layer")
    plt.tight_layout()

    # Save and/or show the plot
    if mode=='s' or mode=='sv':
        if filepath=='0':
            BASE_PATH, _, _, _, _ = path_definition()
            parent_name = model_name[:3]
            short_name = model_name[:-10]
            plt.savefig(f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_activation_violin.png")
        else:
            plt.savefig(filepath)
    if mode=='v' or mode=='sv':
        plt.show()


# To-do
def wt_violin_plot():
    return 0


def activation_box_plot(sampled_files, model, model_name, mode='sv', filepath='0'):
    tf.config.run_functions_eagerly(True)

    # Get list of layer names to analyze
    layers_list = [layer.name for layer in model.layers if isinstance(layer, (Conv2D, Dense))]

    # Collect flattened activations
    layer_activations = {layer_name: [] for layer_name in layers_list}

    for file_path in sampled_files:
        x = load_and_preprocess_image(file_path)
        for layer in model.layers:
            x = layer(x)
            if layer.name in layers_list:
                flat_acts = tf.reshape(x, [-1]).numpy()
                layer_activations[layer.name].append(flat_acts)

    # Concatenate activations per layer
    for layer_name in layer_activations:
        layer_activations[layer_name] = np.concatenate(layer_activations[layer_name])

    # Prepare DataFrame for plotting
    data = []
    for layer_name, acts in layer_activations.items():
        if len(acts) > 10000:  # Optional downsampling for performance
            acts = np.random.choice(acts, 10000, replace=False)
        for val in acts:
            data.append({"Layer": layer_name, "Activation": val})

    df = pd.DataFrame(data)

    # Plot box plot
    plt.figure(figsize=(10, 8))
    sns.boxplot(y="Layer", x="Activation", data=df)
    plt.title("Activation Distributions per Layer (Horizontal Box Plot)")
    plt.xlabel("Activation Value")
    plt.ylabel("Layer")
    plt.tight_layout()

    # Save and/or show the plot
    if mode=='s' or mode=='sv':
        if filepath=='0':
            BASE_PATH, _, _, _, _ = path_definition()
            parent_name = model_name[:3]
            short_name = model_name[:-10]
            plt.savefig(f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_activation_box.png")
        else:
            plt.savefig(filepath)
    if mode=='v' or mode=='sv':
        plt.show()


def wt_box_plot():
    return 0


### Quantization utilities

def wt_range_search(model, model_name, mode='sv', filepath='0'):
    # s: save
    # v: verbose
    # sv: save & verbose
    # Final structured dictionary
    layer_ranges = {}

    for layer in model.layers:
        if hasattr(layer, "get_weights") and hasattr(layer, "set_weights"):
            weights = layer.get_weights()
            if weights:
                sub_dict = {}

                if len(weights) >= 1:  # weights[0] = kernel weights
                    w = np.array(weights[0])
                    sub_dict["weight"] = {
                        "min": float(np.min(w)),
                        "max": float(np.max(w))
                    }

                if len(weights) >= 2:  # weights[1] = biases
                    b = np.array(weights[1])
                    sub_dict["bias"] = {
                        "min": float(np.min(b)),
                        "max": float(np.max(b))
                    }

                layer_ranges[layer.name] = sub_dict

    if mode=='v' or mode=='sv':
        print(json.dumps(layer_ranges, indent=2))

    # Find path
    if mode=='s' or mode=='sv':
        if filepath=='0':
            BASE_PATH, PATH_DATASET, PATH_RAWDATA, PATH_JOINEDDATA, PATH_SAVEDMODELS = path_definition()
            short_name = model_name[:-10]
            range_path = f'{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_wt_range.json'
        else:
            range_path = filepath
        # Save json
        with open(range_path, "w") as f:
            json.dump(layer_ranges, f, indent=4)
        print(f"Saved json in: {range_path}")

    return layer_ranges


def activation_range_search(sampled_files, model, model_name, mode='sv', filepath='0'):
    tf.config.run_functions_eagerly(True)

    # Initialize tracking dict
    layer_min_max = {}
    layers_list = [layer.name for layer in model.layers if isinstance(layer, (Conv2D, Dense))]

    # Process input files
    for file_path in sampled_files:
        x = load_and_preprocess_image(file_path)

        for i, layer in enumerate(model.layers):
            x = layer(x)

            if layer.name in layers_list:
                act_min = float(tf.reduce_min(x).numpy())
                act_max = float(tf.reduce_max(x).numpy())

                if layer.name not in layer_min_max:
                    layer_min_max[layer.name] = {"min": act_min, "max": act_max}
                else:
                    layer_min_max[layer.name]['min'] = float(min(layer_min_max[layer.name]['min'], act_min))
                    layer_min_max[layer.name]['max'] = float(max(layer_min_max[layer.name]['max'], act_max))

    # Save and/or print ranges
    if mode=='s' or mode=='sv':
        if filepath=='0':
            BASE_PATH, _, _, _, _ = path_definition()
            short_name = model_name[:-10]
            filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_activation_range.json"
        # save to json
        with open(filepath, 'w') as f:
            json.dump(layer_min_max, f, indent=4)
    if mode=='v' or mode=='sv':
        for layer_name, stats in layer_min_max.items():
            print(f"{layer_name}: min = {stats['min']:.4f}, max = {stats['max']:.4f}")

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