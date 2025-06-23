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
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from .path import path_definition
from .load_preprocess import load_preprocess
from .model_evaluation import model_evaluation_precise


### Info & visualization for quantization

def subsample_imgdir(num_samples=100):
    # train_dataset, val_dataset = load_preprocess()

    import os
    import random

    dict = path_definition()
    PATH_DATASET = dict['PATH_DATASET']

    classes = [os.path.join(PATH_DATASET, f) for f in os.listdir(PATH_DATASET) if os.path.isdir(os.path.join(PATH_DATASET, f))]

    valid_extensions = ('.png', '.jpg', '.jpeg')

    # Initialize list
    all_files = []

    # Loop through both directories
    for img_dir in classes:
        for f in os.listdir(img_dir):
            full_path = os.path.join(img_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(valid_extensions):
                all_files.append(full_path)
                
    random.seed(99)  # For reproducibility
    sampled_files = random.sample(all_files, num_samples)

    return sampled_files


def load_and_preprocess_image(file_path, img_size=(224, 224, 3)):
    img = tf.keras.utils.load_img(file_path, target_size=img_size)
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


def activation_dist_plots(sampled_files, model, model_name, mode='sv', filepath='0', force=0):
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
                dict = path_definition()
                BASE_PATH = dict['BASE_PATH']
                parent_name = model_name[:3]
                short_name = model_name[:-10]
                tmp_filepath = f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_layer{i:02d}_activation.png"
            else:
                tmp_filepath = f"{filepath}/{short_name}_layer{i:02d}_activation.png"
            if not os.path.exists(tmp_filepath) or force==1:
                plt.savefig(tmp_filepath)
                print(f'Saved plot in {tmp_filepath}')
            else:
                print(f"File already exists in {tmp_filepath}. Change \"force\" arguement to overwrite.")
        if mode=='v' or mode=='sv':
            plt.show()
        else:
            plt.close()


def activation_range_plot(sampled_files, model, model_name, mode='sv', filepath='0', force=0):
    try:
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        testpath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_activation_range.json"
        with open(testpath) as f:
            layer_min_max = json.load(f)
    except:
        layer_min_max = activation_range_search(sampled_files, model, model_name, mode='0', filepath=filepath)

    keys = list(layer_min_max.keys())
    # values = list(layer_min_max.values())
    min_values = []
    max_values = []
    for i in keys:
        # print(layer_min_max[i])
        min_values.append(layer_min_max[i]['min'])
        max_values.append(layer_min_max[i]['max'])

    # Plotting the vector
    plt.plot(keys, max_values, marker='o')  # marker='o' adds dots at data points

    # Adding labels and title (optional)
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.title('Basic Plot of a Vector')

    # Show grid (optional)
    plt.grid(True)

    # Save and/or show the plot
    if mode=='s' or mode=='sv':
        if filepath=='0':
            dict = path_definition()
            BASE_PATH = dict['BASE_PATH']
            parent_name = model_name[:3]
            short_name = model_name[:-10]
            filepath = f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_activation_range.png"
            plt.savefig(filepath)
        if not os.path.exists(filepath) or force==1:
            plt.savefig(filepath)
            print(f'Saved plot in {filepath}')
        else:
            print(f"File already exists in {filepath}. Change \"force\" arguement to overwrite.")
    if mode=='v' or mode=='sv':
        plt.show()


def wt_dist_plots(model, model_name, mode='sv', filepath='0', force=0):
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

    dict = path_definition()
    BASE_PATH = dict['BASE_PATH']
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
            print(f'Saved weight plot in {filepath}')
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
            # Find correct file path
            if filepath=='0':
                tmp_filepath = f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_layer{i:02d}_bias.png"
            else:
                tmp_filepath = f"{filepath}/{short_name}_layer{i:02d}_bias.png"
            # Save if it doesn't exist or force
            if not os.path.exists(tmp_filepath) or force==1:
                plt.savefig(tmp_filepath)
                print(f'Saved plot in {tmp_filepath}')
            else:
                print(f"File already exists in {tmp_filepath}. Change \"force\" arguement to overwrite.")
        if mode=='v' or mode=='sv':
            plt.show()
        plt.close()


def activation_violin_plot(sampled_files, model, model_name, mode='sv', filepath='0', force=0):
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
        # Find correct file path
        if filepath=='0':
            dict = path_definition()
            BASE_PATH = dict['BASE_PATH']
            parent_name = model_name[:3]
            short_name = model_name[:-10]
            filepath = f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_activation_violin.png"
        # Save if it doesn't exist or force
        if not os.path.exists(filepath) or force==1:
            plt.savefig(filepath)
            print(f'Saved plot in {filepath}')
        else:
            print(f"File already exists in {filepath}. Change \"force\" arguement to overwrite.")
    if mode=='v' or mode=='sv':
        plt.show()


def wt_violin_plot(model, model_name, mode='sv', filepath='0', force=0):
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
    # Prepare data for seaborn
    data = []

    for layer_name, weights in weight_distributions.items():
        data.extend([(layer_name + ' - weight', w) for w in weights])

    for layer_name, biases in bias_distributions.items():
        data.extend([(layer_name + ' - bias', b) for b in biases])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Layer', 'Value'])

    # Create violin plot
    plt.figure(figsize=(14, max(6, len(df['Layer'].unique()) * 0.4)))
    sns.violinplot(
        y='Layer',
        x='Value',
        data=df,
        density_norm='width',
        inner='box',
        hue='Layer',
        palette='Set2',
        legend=False
        )

    # Style
    plt.title("Violin Plot of Weight and Bias Distributions per Layer")
    plt.xlabel("Value")
    plt.ylabel("Layer")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if mode=='s' or mode=='sv':
        # Find correct file path
        if filepath=='0':
            dict = path_definition()
            BASE_PATH = dict['BASE_PATH']
            parent_name = model_name[:3]
            short_name = model_name[:-10]
            filepath = f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_wtbias_violin.png"
        # Save if it doesn't exist or force
        if not os.path.exists(filepath) or force==1:
            plt.savefig(filepath)
            print(f'Saved plot in {filepath}')
        else:
            print(f"File already exists in {filepath}. Change \"force\" arguement to overwrite.")
    if mode=='v' or mode=='sv':
        plt.show()
    plt.close()


def activation_box_plot(sampled_files, model, model_name, mode='sv', filepath='0', force=0):
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
        # Find correct file path
        if filepath=='0':
            dict = path_definition()
            BASE_PATH = dict['BASE_PATH']
            parent_name = model_name[:3]
            short_name = model_name[:-10]
            filepath = f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_activation_box.png"
        # Save if it doesn't exist or force
        if not os.path.exists(filepath) or force==1:
            plt.savefig(filepath)
            print(f'Saved plot in {filepath}')
        else:
            print(f"File already exists in {filepath}. Change \"force\" arguement to overwrite.")
    if mode=='v' or mode=='sv':
        plt.show()


def wt_box_plot(model, model_name, mode='sv', filepath='0', force=0):
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
    # Prepare data for seaborn
    data = []
    def downsample(values, max_len=1000):
        values = list(values)  # Convert numpy array to list
        return random.sample(values, min(len(values), max_len))

    # Prepare data
    data = []

    for layer_name, weights in weight_distributions.items():
        ws = downsample(weights)
        data.extend([(f'{layer_name} - weight', w) for w in ws])

    for layer_name, biases in bias_distributions.items():
        bs = downsample(biases)
        data.extend([(f'{layer_name} - bias', b) for b in bs])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Layer', 'Value'])

    # Plot
    plt.figure(figsize=(14, max(6, len(df['Layer'].unique()) * 0.4)))
    ax = sns.boxplot(y='Layer', x='Value', data=df, hue='Layer', palette='Set3', fliersize=2)

    legend = ax.get_legend()
    if legend:
        legend.remove()

    # Style
    plt.title("Boxplot of Weight and Bias Distributions per Layer")
    plt.xlabel("Value")
    plt.ylabel("Layer")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if mode=='s' or mode=='sv':
        # Find correct file path
        if filepath=='0':
            dict = path_definition()
            BASE_PATH = dict['BASE_PATH']
            parent_name = model_name[:3]
            short_name = model_name[:-10]
            filepath = f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_wtbias_box.png"
        # Save if it doesn't exist or force
        if not os.path.exists(filepath) or force==1:
            plt.savefig(filepath)
            print(f'Saved plot in {filepath}')
        else:
            print(f"File already exists in {filepath}. Change \"force\" arguement to overwrite.")
    if mode=='v' or mode=='sv':
        plt.show()


def wt_histogram_ranges(model, model_name, mode='sv', filepath='0', force=0):
    layer_ranges = wt_range_search(model, model_name, mode='0')


    # Prepare data for seaborn
    data = []
    def downsample(values, max_len=1000):
        values = list(values)  # Convert numpy array to list
        return random.sample(values, min(len(values), max_len))

    # Prepare data
    layers = []
    weight_mins = []
    weight_maxs = []
    bias_mins = []
    bias_maxs = []

    for layer_name, data in layer_ranges.items():
        layers.append(layer_name)
        weight_mins.append(data.get("weight", {}).get("min", np.nan))
        weight_maxs.append(data.get("weight", {}).get("max", np.nan))
        bias_mins.append(data.get("bias", {}).get("min", np.nan))
        bias_maxs.append(data.get("bias", {}).get("max", np.nan))

    indices = np.arange(len(layers))
    bar_height = 0.35

    plt.figure(figsize=(12, 0.6 * len(layers)))

    layers = layers[::-1]
    weight_mins = weight_mins[::-1]
    weight_maxs = weight_maxs[::-1]
    bias_mins = bias_mins[::-1]
    bias_maxs = bias_maxs[::-1]

    # Plot weight ranges
    plt.barh(indices + bar_height/2, np.array(weight_maxs) - np.array(weight_mins),
            left=weight_mins, height=bar_height, label='Weight Range', color='skyblue', edgecolor='black')

    # Plot bias ranges
    plt.barh(indices - bar_height/2, np.array(bias_maxs) - np.array(bias_mins),
            left=bias_mins, height=bar_height, label='Bias Range', color='salmon', edgecolor='black')

    # Labels and style
    plt.yticks(indices, layers)
    plt.xlabel("Value Range")
    plt.title("Weight and Bias Ranges per Layer")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    if mode=='s' or mode=='sv':
        # Find correct file path
        if filepath=='0':
            dict = path_definition()
            BASE_PATH = dict['BASE_PATH']
            parent_name = model_name[:3]
            short_name = model_name[:-10]
            filepath = f"{BASE_PATH}/Docs_Reports/AnalysisPlots/{parent_name}/{short_name}_wtbias_hist.png"
        # Save if it doesn't exist or force
        if not os.path.exists(filepath) or force==1:
            plt.savefig(filepath)
            print(f'Saved plot in {filepath}')
        else:
            print(f"File already exists in {filepath}. Change \"force\" arguement to overwrite.")
    if mode=='v' or mode=='sv':
        plt.show()


### Quantization utilities (Helper functions)

def find_smallest_power_of_two(a, b):
    """
    Finds the smallest positive integer k such that a * 2**k > b.
    Returns (k, a * 2**k).
    """
    if a > b:
        raise ValueError("a must be smaller than b")

    k = 1
    while a * 2**k <= b:
        k += 1
    return k, a * 2**k


def find_biggest_power_of_two(a, b):
    if a < b:
        raise ValueError("a must be bigger than b")
    
    k = 0
    while a / (2 ** (k + 1)) > b:
        k += 1
    result = a / (2 ** k)
    return k, result


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


def compute_symmetric_int8_wt_scales(range_dict):
    # Helper function for wt_scale_search
    scale_dict = {}
    for layer_name, layer_data in range_dict.items():
        if "weight" in layer_data:
            w_min = layer_data["weight"]["min"]
            w_max = layer_data["weight"]["max"]
            max_abs = max(abs(w_min), abs(w_max))
            scale = max_abs / 127 # scale = r/q
            scale_dict[layer_name] = scale
    return scale_dict


def gen_sample_paths(path_dataset='0', num_samples=500):
    import os
    import random

    if(path_dataset=='0'):
        dict = path_definition()
        directory = dict['PATH_DATASET']
    else:
        directory = path_dataset
    class_path = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    img_size = (224, 224, 3)
    num_samples = num_samples
    valid_extensions = ('.png', '.jpg', '.jpeg')

    # Initialize list
    all_files = []

    # Loop through both directories
    for img_dir in class_path:
        for f in os.listdir(img_dir):
            full_path = os.path.join(img_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(valid_extensions):
                all_files.append(full_path)
            
    random.seed(99)  # For reproducibility
    sampled_files = random.sample(all_files, min(num_samples, len(all_files)))

    return sampled_files


### Quantization utilities

def input_range_search(mode='v', num_samples=300, filepath='0',force=0):
    # v: prints input range
    # force: 0 -> if file exists read from path & return, if it doesn't exist calculate & write, return
    #        1 -> calculate &  return, if file exists, ask if you want to overwrite

    if(filepath == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        tmp_filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/input_range.json"
    else:
        tmp_filepath = filepath

    ask_message = 0
    if(force==0):
        if os.path.exists(tmp_filepath):
            try:
                with open(tmp_filepath, 'r') as f:
                    input_dict = json.load(f)
                print(f'Read input range json from {tmp_filepath}')
            except:
                print('Wrong format for reading input range from json!!')
            calculate = 0
            # revoke save mode
            if(mode == 's'):
                mode = ''
            if(mode == 'sv'):
                mode = 'v'
        else:
            calculate = 1
    else:
        calculate = 1
        if os.path.exists(tmp_filepath):
            ask_message = 1

    if(calculate == 1):
        if(mode=='v' or mode=='sv'):
            print('Calculating the Input Range...')
        
        global_min = tf.constant(float('inf'))
        global_max = tf.constant(float('-inf'))

        sampled_files = subsample_imgdir(num_samples=num_samples)

        for file_path in sampled_files:
            input_img = load_and_preprocess_image(file_path)
            batch_min = tf.reduce_min(input_img)
            batch_max = tf.reduce_max(input_img)
            global_min = tf.minimum(global_min, batch_min)
            global_max = tf.maximum(global_max, batch_max)

        input_dict = {}
        input_dict['input_layer'] = {'min': float(global_min.numpy()), 'max': float(global_max.numpy())}

    if(mode=='v' or mode=='sv'):
        if(calculate == 1):
            print(f"Input tensor range over {len(sampled_files)} images:")
        print(f"min = {input_dict['input_layer']['min']}, max = {input_dict['input_layer']['max']}")
    # Shows message for the user to choose if they want to overwrite
    if(ask_message==1 and (mode == 's' or mode == 'sv')):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
                break
            else:
                print("Invalid input.")
    if(mode=='s' or mode=='sv'):
        parent_folder = os.path.dirname(tmp_filepath)
        os.makedirs(parent_folder, exist_ok=True)
        with open(tmp_filepath, "w") as f:
            json.dump(input_dict, f, indent=4)
        print(f"Saved json in: {tmp_filepath}") 

    return input_dict['input_layer']['min'], input_dict['input_layer']['max']


def wt_range_search(model, model_name, mode='sv', filepath='0', force=0):
    # s: save
    # v: verbose
    # sv: save & verbose
    # force: 0 -> if file exists read from path & return, if it doesn't exist calculate & write, return
    #        1 -> calculate &  return, if file exists, ask if you want to overwrite

    # Find path
    if(filepath == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        tmp_filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_wt_range.json"
    else:
        tmp_filepath = filepath

    ask_message = 0
    if(force==0):
        if os.path.exists(tmp_filepath):
            try:
                with open(tmp_filepath, 'r') as f:
                    layer_ranges = json.load(f)
                print(f'Read weight range json from {tmp_filepath}')
                calculate = 0
            except:
                print('Wrong format for reading wt range from json!!')
                calculate = 1
            # revoke save mode
            if(mode == 's'):
                mode = ''
            if(mode == 'sv'):
                mode = 'v'
        else:
            calculate = 1
    else:
        calculate = 1
        if os.path.exists(tmp_filepath):
            ask_message = 1

    if(calculate == 1):
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

    # Shows message for the user to choose if they want to overwrite
    if(ask_message==1 and (mode == 's' or mode == 'sv') and BASE_PATH[:7]!='/kaggle'):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
                break
            else:
                print("Invalid input.")

    if mode=='s' or mode=='sv':
        parent_folder = os.path.dirname(tmp_filepath)
        os.makedirs(parent_folder, exist_ok=True)
        with open(tmp_filepath, "w") as f:
            json.dump(layer_ranges, f, indent=4)
        print(f"Saved json in: {tmp_filepath}")
    return layer_ranges


def activation_range_search(sampled_files, model, model_name, mode='sv', filepath='0', force=0):
    # s: save
    # v: verbose
    # sv: save & verbose
    # force: 0 -> if file exists read from path & return, if it doesn't exist calculate & write, return
    #        1 -> calculate &  return, if file exists, ask if you want to overwrite

    tf.config.run_functions_eagerly(True)

    # Find path
    if(filepath == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        tmp_filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_activation_range.json"
    else:
        tmp_filepath = filepath
    
    ask_message = 0
    if(force==0):
        if os.path.exists(tmp_filepath):
            try:
                with open(tmp_filepath, 'r') as f:
                    range_serializable = json.load(f)
                print(f'Read activation range json from {tmp_filepath}')
            except:
                print('Wrong format for reading activation sw range from json!!')
            calculate = 0
            # revoke save mode
            if(mode == 's'):
                mode = ''
            if(mode == 'sv'):
                mode = 'v'
        else:
            calculate = 1
    else:
        calculate = 1
        if os.path.exists(tmp_filepath):
            ask_message = 1

    if(calculate == 1):
        # Initialize tracking dict
        layer_min_max = {}
        layers_list = ['input_layer']
        for layer in model.layers:
            if isinstance(layer, (Conv2D, Dense)):
                layers_list.append(layer.name)

        in_min, in_max = input_range_search(mode=mode)
        layer_min_max[layers_list[0]] = {"min": -max(abs(in_min), in_max), "max": max(abs(in_min), in_max)}

        # Process input files
        for file_path in sampled_files:
            x = load_and_preprocess_image(file_path)

            for i, layer in enumerate(model.layers):
                x = layer(x)

                if layer.name in layers_list:
                    # act_min = tf.reduce_min(x).numpy()
                    act_max = tf.reduce_max(x).numpy()

                    if layer.name not in layer_min_max:
                        # layer_min_max[layer.name] = {"min": act_min, "max": act_max}
                        layer_min_max[layer.name] = {"min": 0, "max": act_max}
                    else:
                        # layer_min_max[layer.name]['min'] = min(layer_min_max[layer.name]['min'], act_min)
                        layer_min_max[layer.name]['max'] = max(layer_min_max[layer.name]['max'], act_max)
        
        range_serializable = {
            layer: {
                "min": float(stats["min"]),
                "max": float(stats["max"])
            }
            for layer, stats in layer_min_max.items()
        }

    if mode=='v' or mode=='sv':
        for layer_name, stats in range_serializable.items():
            print(f"{layer_name}: min = {stats['min']:.4f}, max = {stats['max']:.4f}")

    # Shows message for the user to choose if they want to overwrite
    if(ask_message==1 and (mode == 's' or mode == 'sv') and BASE_PATH[:7]!='/kaggle'):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
                break
            else:
                print("Invalid input.")

    # Save and/or print ranges
    if mode=='s' or mode=='sv':
        if filepath=='0':
            dict = path_definition()
            BASE_PATH = dict['BASE_PATH']
            short_name = model_name[:-10]
            filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_activation_range.json"        
        # Save json if it doesn't exist or force
        if not os.path.exists(filepath) or force==1:
            parent_folder = os.path.dirname(filepath)
            os.makedirs(parent_folder, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(range_serializable, f, indent=4)
            print(f'Saved activation ranges in {filepath}')
        else:
            print(f"File already exists in {filepath}. Change \"force\" arguement to overwrite.")

    return range_serializable


def wt_hw_range_search(model_name, activation_range_dict, wt_range_dict, filepath='0', force=0,  mode='sv', debug=0, num_bits=8):
    # s: save
    # v: verbose
    # sv: save & verbose
    # force: 0 -> if file exists read from path & return, if it doesn't exist calculate & write, return
    #        1 -> calculate &  return, if file exists, ask if you want to overwrite
    # debug: prints layer by layer calculations

    # Find correct path
    if(filepath == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        tmp_filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_wt_hw_range.json"
    else:
        tmp_filepath = filepath

    # Flag inputs handling
    ask_message = 0
    if os.path.exists(tmp_filepath):
        try:
            with open(tmp_filepath, 'r') as f:
                wt_hw_range_dict = json.load(f)
                if f"{num_bits}b" in wt_hw_range_dict and wt_hw_range_dict[f"{num_bits}b"]:
                    print(f'Read wt_hw_range json dictionary from {tmp_filepath} and it has values for {num_bits} bits.')
                    if force==1:
                        calculate = 1
                        ask_message = 1
                    else:
                        calculate = 0
                        # revoke save mode
                        if(mode == 's'):
                            mode = ''
                        if(mode == 'sv'):
                            mode = 'v'
                else:
                    print(f"'{num_bits}b' is missing or empty from dictionary.")
                    calculate = 1
        except:
            print('Wrong format for reading wt_hw_range json!!')
            calculate = 1
            wt_hw_range_dict = {}
    else:
        calculate = 1
        wt_hw_range_dict = {}

    if(calculate==1 and force==1):
        ask_message = 1
    else:
        ask_message = 0

    if mode=='v' or mode=='sv':
        verbose = 1
    else:
        verbose = 0

    bw_range_dict = {}
    if(calculate == 1):
        layer_list = list(activation_range_dict.keys())
        if(verbose==1):
            print(layer_list)
            print('\n')

        for i in range(1, len(layer_list)):
            if(verbose==1 or debug==1):
                print(f'For layer {i}.')
        
            tmp = activation_range_dict[layer_list[i]]['max'] * (2**(num_bits-1)-1)/activation_range_dict[layer_list[i-1]]['max']

            # find biggest positive integer k, so that tmp * 2**k is bigger than the max weight range
            k, wt_range = find_biggest_power_of_two(tmp, wt_range_dict[layer_list[i]]['weight']['max'])

            N = num_bits + k

            bw_range_dict[layer_list[i]] = {"min": float(-wt_range), "max": float(wt_range)}

            if(debug==1):
                print(f'tmp: {tmp}')
                print(f'Input: {activation_range_dict[layer_list[i-1]]}')
                print(f'Next input: {activation_range_dict[layer_list[i]]}')
            if(verbose==1):
                print(f"Weight range: {wt_range_dict[layer_list[i]]['weight']['max']}")
                print(f"HW weight range: {wt_range}")
            if(debug==1):
                print(f'Accumulator bitwidth {N}')
                print(f'Precision bitwidth {num_bits}')
                print(f'Shift right by {k}')
            if(verbose==1 or debug==1):
                print('\n')

            # Update or add the subdictionary for give bit-width
            wt_hw_range_dict[f"{num_bits}b"] = bw_range_dict

    # Shows message for the user to choose if they want to overwrite
    if(ask_message==1 and (mode == 's' or mode == 'sv') and BASE_PATH[:7]!='/kaggle'):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
                break
            else:
                print("Invalid input.")

    if mode=='s' or mode=='sv':
        parent_folder = os.path.dirname(tmp_filepath)
        os.makedirs(parent_folder, exist_ok=True)
        with open(tmp_filepath, "w") as f:
            json.dump(wt_hw_range_dict, f, indent=4)
        print(f"Saved weight HW range dictionary json in: {tmp_filepath}")

    return wt_hw_range_dict


def activation_hw_range_search(model_name, activation_range_dict, wt_range_dict, filepath='0', force=0, mode='sv', debug=0, num_bits=8):
    # s: save
    # v: verbose
    # sv: save & verbose
    # force: 0 -> if file exists read from path & return, if it doesn't exist calculate & write, return
    #        1 -> calculate &  return, if file exists, ask if you want to overwrite
    # debug: prints layer by layer calculations

    # Find correct path
    if(filepath == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        tmp_filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_activation_hw_range.json"
    else:
        tmp_filepath = filepath
    
    # Flag inputs handling
    ask_message = 0
    if os.path.exists(tmp_filepath):
        try:
            with open(tmp_filepath, 'r') as f:
                activation_hw_range_dict = json.load(f)
                if f"{num_bits}b" in activation_hw_range_dict and activation_hw_range_dict[f"{num_bits}b"]:
                    print(f'Read activation_hw_range json dictionary from {tmp_filepath} and it has values for {num_bits} bits.')
                    if force==1:
                        calculate = 1
                        ask_message = 1
                    else:
                        calculate = 0
                else:
                    print(f"'{num_bits}b' is missing or empty from dictionary.")
                    calculate = 1
        except:
            print('Wrong format for reading complete dictionarys json!!')
            calculate = 1
            activation_hw_range_dict = {}
    else:
        calculate = 1
        activation_hw_range_dict = {}

    if mode=='v' or mode=='sv':
        verbose = 1
    else:
        verbose = 0

    if(calculate == 1):
        layer_list = list(activation_range_dict.keys())
        if(verbose==1):
            print(layer_list)
            print('\n')

        bw_range_dict = {}
        in_min, in_max = input_range_search(mode='')
        bw_range_dict[layer_list[0]] = {"min": -max(abs(in_min), in_max), "max": max(abs(in_min), in_max)}

        for i in range(1, len(layer_list)):
            if(verbose==1 or debug==1):
                print(f'For layer {i}.')

            tmp = activation_range_dict[layer_list[i-1]]['max'] * wt_range_dict[layer_list[i]]['weight']['max'] / (2**(num_bits-1)-1)
            
            k, activation_range = find_smallest_power_of_two(tmp, activation_range_dict[layer_list[i]]['max'])
            
            N = num_bits + k

            bw_range_dict[layer_list[i]] = {"min": 0, "max": float(activation_range)}

            if(debug==1):
                print(f'tmp: {tmp}')
                print(f'Input: {activation_range_dict[layer_list[i-1]]}')
                print(f"Weight range: {wt_range_dict[layer_list[i]]['weight']['max']}")
            if(verbose==1):
                print(f'Next input range: {activation_range_dict[layer_list[i]]}')
                print(f"HW next input range: {activation_range}")
            if(debug==1):
                print(f'Accumulator bitwidth {N}')
                print(f'Precision bitwidth {num_bits}')
                print(f'Shift right by {k}')
            if(verbose==1 or debug==1):
                print('\n')

            # Update or add the subdictionary for give bit-width
            activation_hw_range_dict[f"{num_bits}b"] = bw_range_dict

    if (mode=='v' or mode=='sv') and calculate==0:
        print(json.dumps(activation_hw_range_dict, indent=4))

    # Shows message for the user to choose if they want to overwrite
    if(ask_message==1 and (mode == 's' or mode == 'sv')):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
                break
            else:
                print("Invalid input.")

    if mode=='s' or mode=='sv':
        parent_folder = os.path.dirname(tmp_filepath)
        os.makedirs(parent_folder, exist_ok=True)
        with open(tmp_filepath, "w") as f:
            json.dump(activation_hw_range_dict, f, indent=4)
        print(f"Saved activation_hw_range_dict json in: {tmp_filepath}")

    return activation_hw_range_dict


def activation_hw_range_search_old(model_name, activation_sw_range_dict, activation_sw_scale_dict, wt_range_dict, wt_scale_dict, debug=0, force=0, filepath='0', mode='sv', num_bits=8, precision='uint'):
    # s: save
    # v: verbose
    # sv: save & verbose
    # force: 0 -> if file exists read from path & return, if it doesn't exist calculate & write, return
    #        1 -> calculate &  return, if file exists, ask if you want to overwrite
    # debug: prints layer by layer calculations

    # Find correct path
    if(filepath == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        tmp_filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_complete_dict.json"
    else:
        tmp_filepath = filepath
    
    # Flag inputs handling
    ask_message = 0
    if(force==0):
        if os.path.exists(tmp_filepath):
            try:
                with open(tmp_filepath, 'r') as f:
                    complete_dict = json.load(f)
                print(f'Read complete json dictionary from {tmp_filepath}')
            except:
                print('Wrong format for reading complete dictionarys json!!')
            calculate = 0
            # revoke save mode
            if(mode == 's'):
                mode = ''
            if(mode == 'sv'):
                mode = 'v'
        else:
            calculate = 1
    else:
        calculate = 1
        if os.path.exists(tmp_filepath):
            ask_message = 1

    if mode=='v' or mode=='sv':
        verbose = 1
    else:
        verbose = 0

    if(calculate == 1):
        layer_list = list(activation_sw_range_dict.keys())
        if(verbose==1):
            print(layer_list)
            print('\n')

        # scale = activation_sw_scale_dict[layer_list[0]]
        if (precision=='uint'):
            scale = activation_sw_range_dict[layer_list[0]]['max'] / (2**(num_bits)-1)
        elif (precision=='int'):
            scale = activation_sw_range_dict[layer_list[0]]['max'] / (2**(num_bits-1)-1)
        else:
            ValueError('Wrong precision has been given to activation_hw_range_search()')

        # Initialize dictionaries
        activation_hw_scale_dict = {}
        activation_hw_scale_dict[layer_list[0]] = scale
        activation_hw_range_dict = {}
        activation_hw_range_dict[layer_list[0]] = {'min': activation_sw_range_dict[layer_list[0]]['min'],
                                                'max': activation_sw_range_dict[layer_list[0]]['max']}
        activation_shift_dict = {}
        activation_shift_dict[layer_list[0]] = 0

        for i in range(1, len(layer_list)):
            if(verbose==1 or debug==1):
                print(f'For layer {i}.')
            scale_prev = scale
            scale_accumulator = scale_prev * wt_scale_dict[layer_list[i]]
            quant_max = activation_sw_range_dict[layer_list[i]]['max'] / scale_accumulator # q = r/scale
            if (precision=='int'):
                quant_max = 2*quant_max

            quant_exp = math.ceil(math.log2(quant_max))
            quant_poweroftwo = 2 ** quant_exp
            shift = quant_exp - num_bits
            scale = scale_accumulator*(2**shift)
            if (precision=='uint'):
                hw_max = ((2**(num_bits)-1)) * scale  # r = q*scale
            elif (precision=='int'):
                hw_max = ((2**(num_bits-1)-1)) * scale  # r = q*scale
            else:
                ValueError('Wrong precision input!')

            if(verbose==1):
                print(f'Scale accumulator: {scale_accumulator}')
                print(f'Scale: {scale}')
                print(f"Previous layer activation: {activation_sw_range_dict[layer_list[i-1]]['max']}")
                print(f"Activation: {activation_sw_range_dict[layer_list[i]]['max']}")
            if(debug==1):
                print(f'Quant_max: {quant_max}')
                print(f'Quant_exp: {quant_exp}, for both signs.')
                print(f'Quant_poweroftwo: {quant_poweroftwo}, for both signs.')
                print(f'Layer {i}: scale ratio:{(2**shift)}')
            if(verbose==1):
                print(f'Hw_max: {hw_max}')
                print(f'Shift result by {shift}')
            if(verbose==1 or debug==1):
                print('\n')
            
            activation_hw_scale_dict[layer_list[i]] = scale
            activation_hw_range_dict[layer_list[i]] = {'min': 0, 'max': hw_max}
            activation_shift_dict[layer_list[i]] = shift

        complete_dict = {
            'activation_hw_scale': activation_hw_scale_dict,
            'activation_sw_scale': activation_sw_scale_dict,
            'wt_scale': wt_scale_dict,
            'activation_hw_range_dict': activation_hw_range_dict,
            'activation_sw_range_dict': activation_sw_range_dict,
            'wt_range': wt_range_dict,
            'shift': activation_shift_dict
        }

    if (mode=='v' or mode=='sv') and calculate==0:
        print(json.dumps(complete_dict, indent=4))

    # Shows message for the user to choose if they want to overwrite
    if(ask_message==1 and (mode == 's' or mode == 'sv')):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
                break
            else:
                print("Invalid input.")

    if mode=='s' or mode=='sv':
        # Changed if I want to save separate json files
        # with open(tmp_filepath, "w") as f:
        #     json.dump(activation_hw_scale_dict, f, indent=4)
        # print(f"Saved hw_scale json in: {tmp_filepath}")
        # with open(tmp_filepath, "w") as f:
        #     json.dump(activation_hw_range_dict, f, indent=4)
        # print(f"Saved hw_range json in: {tmp_filepath}")
        # with open(tmp_filepath, "w") as f:
        #     json.dump(activation_shift_dict, f, indent=4)
        # print(f"Saved shift json in: {tmp_filepath}")
        parent_folder = os.path.dirname(tmp_filepath)
        os.makedirs(parent_folder, exist_ok=True)
        with open(tmp_filepath, "w") as f:
            json.dump(complete_dict, f, indent=4)
        print(f"Saved complete_dict json in: {tmp_filepath}")

    return complete_dict


def complete_dict_search(model, model_name, filepath='0', force=0, mode='sv', debug=0, num_bits=8):
    # s: save
    # v: verbose
    # sv: save & verbose
    # force: 0 -> if file exists read from path & return, if it doesn't exist calculate & write, return
    #        1 -> calculate &  return, if file exists, ask if you want to overwrite
    # debug: prints layer by layer calculations

    if(filepath == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        filepath = f'{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_complete_dict.json'
    else:
        filepath = filepath

    ask_message = 0
    if(force==0):
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    complete_dict = json.load(f)
                print(f'Read complete dictionary json from {filepath}')
                calculate = 0
                 # revoke save mode
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
            except:
                print('Wrong format for reading complete dictionary from json!!')
                calculate = 1
        else:
            calculate = 1
    else:
        calculate = 1
        if os.path.exists(filepath):
            ask_message = 1

    if (calculate == 1):
        sampled_files = gen_sample_paths()
        activation_range_dict = activation_range_search(sampled_files, model, model_name, force=force, mode=mode)
        wt_range_dict = wt_range_search(model, model_name, force=force, mode=mode)

        activation_hw_range_dict = activation_hw_range_search(model_name, activation_range_dict, wt_range_dict, force=force, debug=debug, num_bits=num_bits)

        wt_hw_range_dict = wt_hw_range_search(model_name, activation_range_dict, wt_range_dict, force=force, debug=debug, num_bits=num_bits)

        # with open(tmp_filepath, 'r') as f:
        #     wt_hw_range_dict = json.load(f)

        complete_dict = {"activation_range": activation_range_dict,
                        "wt_range": wt_range_dict,
                        "activation_hw_range": activation_hw_range_dict,
                        "wt_hw_range": wt_hw_range_dict}
    
    if mode=='v' or mode=='sv':
        print(json.dumps(complete_dict, indent=2))

    # Shows message for the user to choose if they want to overwrite
    if(ask_message==1 and (mode == 's' or mode == 'sv')):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
                break
            else:
                print("Invalid input.")

    if mode=='s' or mode=='sv':
        parent_folder = os.path.dirname(filepath)
        os.makedirs(parent_folder, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(complete_dict, f, indent=4)
        print(f"Saved json in: {filepath}")

    return complete_dict


### To be deprecated

def wt_scale_search(wt_range_dict, model_name, filepath='0', force=0, mode='sv'):
    # s: save
    # v: verbose
    # sv: save & verbose
    # force: 0 -> if file exists read from path & return, if it doesn't exist calculate & write, return
    #        1 -> calculate &  return, if file exists, ask if you want to overwrite

    if(filepath == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        tmp_filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_wt_scale.json"
    else:
        tmp_filepath = filepath
    
    ask_message = 0
    if(force==0):
        if os.path.exists(tmp_filepath):
            try:
                with open(tmp_filepath, 'r') as f:
                    wt_scale_dict = json.load(f)
                print(f'Read weight scale json from {tmp_filepath}')
            except:
                print('Wrong format for reading wt scale from json!!')
            calculate = 0
            # revoke save mode
            if(mode == 's'):
                mode = ''
            if(mode == 'sv'):
                mode = 'v'
        else:
            calculate = 1
    else:
        calculate = 1
        if os.path.exists(tmp_filepath):
            ask_message = 1
    if(calculate == 1):
        wt_scale_dict = compute_symmetric_int8_wt_scales(wt_range_dict)

    if mode=='v' or mode=='sv':
        for layer, scale in wt_scale_dict.items():
            print(f"{layer}: scale = {scale:.8f}")

    # Shows message for the user to choose if they want to overwrite
    if(ask_message==1 and (mode == 's' or mode == 'sv')):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
                break
            else:
                print("Invalid input.")

    if mode=='s' or mode=='sv':
        parent_folder = os.path.dirname(tmp_filepath)
        os.makedirs(parent_folder, exist_ok=True)
        with open(tmp_filepath, "w") as f:
            json.dump(wt_scale_dict, f, indent=4)
        print(f"Saved json in: {tmp_filepath}")

    return wt_scale_dict


def compute_activation_scales(range_dict, num_bits=8, precision='uint'):
    # q = r/scale
    scale_dict = {}
    for layer_name, layer_data in range_dict.items():
        w_min = layer_data["min"]
        w_max = layer_data["max"]
        max_abs = max(abs(w_min), abs(w_max))
        if (precision=='uint'):
            scale = max_abs / (2**(num_bits)-1) # scale = r/q
        elif (precision=='int'):
            scale = max_abs / (2**(num_bits-1)-1) # scale = r/q
        else:
            ValueError('Wrong precision input!')
        scale_dict[layer_name] = scale
    return scale_dict


def activation_sw_scale_search(activation_sw_range_dict, model_name, filepath='0', force=0, mode='sv', num_bits=8, precision='uint'):
    # s: save
    # v: verbose
    # sv: save & verbose
    # force: 0 -> if file exists read from path & return, if it doesn't exist calculate & write, return
    #        1 -> calculate &  return, if file exists, ask if you want to overwrite
    if(filepath == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        tmp_filepath = f"{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_activation_sw_scale_{precision}.json"
    else:
        tmp_filepath = filepath

    ask_message = 0
    if(force==0):
        if os.path.exists(tmp_filepath):
            try:
                with open(tmp_filepath, 'r') as f:
                    activation_sw_scale_dict = json.load(f)
                print(f'Read activation sw scale json from {tmp_filepath}')
            except:
                print('Wrong format for reading activation sw scale from json!!')
            calculate = 0
            # revoke save mode
            if(mode == 's'):
                mode = ''
            if(mode == 'sv'):
                mode = 'v'
        else:
            calculate = 1
    else:
        calculate = 1
        if os.path.exists(tmp_filepath):
            ask_message = 1
    if(calculate == 1):
        activation_sw_scale_dict = compute_activation_scales(activation_sw_range_dict, num_bits=num_bits, precision=precision)

    if mode=='v' or mode=='sv':
        for layer, scale in activation_sw_scale_dict.items():
            print(f"{layer}: scale = {scale:.8f}")

    # Shows message for the user to choose if they want to overwrite
    if(ask_message==1 and (mode == 's' or mode == 'sv')):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                if(mode == 's'):
                    mode = ''
                if(mode == 'sv'):
                    mode = 'v'
                break
            else:
                print("Invalid input.")

    if mode=='s' or mode=='sv':
        parent_folder = os.path.dirname(tmp_filepath)
        os.makedirs(parent_folder, exist_ok=True)        
        with open(tmp_filepath, "w") as f:
            json.dump(activation_sw_scale_dict, f, indent=4)
        print(f"Saved json in: {tmp_filepath}")

    return activation_sw_scale_dict


### Quantize models & evaluate

def quant_activations(model, model_name, num_bits=8, input_shape=(224,224,3), mode='eval', range_path='0', design='sw', batch_len=157):
    # quant: returns model with quantized weights
    # eval: evaluates model with quantized weights & returns model with quantized weights    
    # 'sw' means quantization is run based on arbitrary symmetric ranges of max values
    # 'hw' means hw efficient quantization is run, so that scales from previous to next layer are only calculated based on shifting bits

    # Show mode message
    if(design=='sw'):
        print('Quantization on arbitrary symmetric ranges is applied.')
    elif(design=='hww'):
        print('Quantization on symmetric ranges that enable shifting on interlayer scaling is applied.')
        print('Weight focused solution chosen.')
    elif(design=='hwa'):
        print('Quantization on symmetric ranges that enable shifting on interlayer scaling is applied.')
        print('Activation focused solution chosen.')
    else:
        ValueError("Wrong design variable input.")

    # Read appropriate ranges
    if(range_path == '0'):
        dict = path_definition()
        BASE_PATH = dict['BASE_PATH']
        short_name = model_name[:-10]
        if(design=='hww'):
            filepath_var = '_hw_'
        else:
            filepath_var = '_'
        filepath = f'{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_activation{filepath_var}range.json'
    else:
        filepath = range_path

    try:
        with open(filepath, 'r') as f:
            range_dict_tmp = json.load(f)
        if(design=='hww'):
            range_dict = range_dict_tmp[f"{num_bits}b"]
        else:
            range_dict = range_dict_tmp
        print(f'Read {design} activation quantization range from {filepath}.')
    except:
        print(f'Quantization range not found in {filepath}, recalculating.')
        # calculate and save json with ranges
        if(design=='hww'):
            complete_dict = complete_dict_search(model, model_name, force=0, mode='s', debug=0, num_bits=num_bits)
            range_dict = complete_dict["activation_hw_range"][f"{num_bits}b"]
        else:
            sampled_files = gen_sample_paths()
            range_dict = activation_range_search(sampled_files, model, model_name, mode='s')
        # Save json
        try:
            with open(filepath, 'w') as f:
                json.dump(range_dict, f, indent=4)
            print(f'Quantization range saved in {filepath}!')
        except:
            print(f'Quantization range could not be saved in {filepath}!')

    # quant model and evaluate
    quant_activation_model = clone_model_with_fake_quant(model, input_shape, range_dict, num_bits=num_bits)
    quant_activation_model.compile(optimizer='adam', loss=model.loss, metrics=['accuracy'])

    if (mode=='eval'):
        acc, loss = model_evaluation_precise(quant_activation_model, batch_len=batch_len, mode='test')
    else:
        acc = None
        loss = None

    return quant_activation_model, acc, loss


def quant_weights(model, model_name, num_bits=8, quant='symmetric', mode='eval', design='sw', batch_len=157):
    # quant: returns model with quantized weights
    # eval: evaluates model with quantized weights & returns model with quantized weights
    if(quant!='symmetric'):
        print('No asymmetric quantization developed yet!')
        return 1
    
    # if hwa -> activation_range & wt_hw_range
    # if hww -> activation_hw_range & wt_range
    # if sw -> activation_range & wt_range

    # Get activation & wt ranges from json or calculate them
    if design == 'sw':
        sampled_files = gen_sample_paths()
        activation_range_dict = activation_range_search(sampled_files, model, model_name, mode='s', force=0)
        wt_range_dict = wt_range_search(model, model_name, mode='s', force=0)
    elif design == 'hww':
        sampled_files = gen_sample_paths()
        wt_range_dict = wt_range_search(model, model_name, mode='s', force=0)
        activation_range_dict_tmp = activation_range_search(sampled_files, model, model_name, mode='s', filepath='0', force=0)
        activation_range_dict_tmp2 = activation_hw_range_search(model_name, activation_range_dict_tmp, wt_range_dict, force=0, debug=0, num_bits=num_bits)
        activation_range_dict = activation_range_dict_tmp2[f"{num_bits}b"]
    elif design == 'hwa':
        sampled_files = gen_sample_paths()
        activation_range_dict = activation_range_search(sampled_files, model, model_name, mode='s', filepath='0', force=0)
        wt_range_dict_tmp = wt_range_search(model, model_name, mode='s', force=0)
        wt_range_dict_tmp2 = wt_hw_range_search(model_name, activation_range_dict, wt_range_dict_tmp, force=0, debug=0, num_bits=num_bits)
        wt_range_dict = wt_range_dict_tmp2[f"{num_bits}b"]
    else:
        ValueError("Wrong design variable input.")

    # activation_range_filepath = f'{BASE_PATH}/Docs_Reports/Quant/Ranges/{short_name}_activation_{design}_range.json'
    # with open(activation_range_filepath, 'r') as f:
    #     activation_ranges = json.load(f)


    # # This would quantize biases based on their range
    # for layer in model.layers:
    #     if hasattr(layer, "get_weights") and hasattr(layer, "set_weights"):
    #         weights = layer.get_weights()
    #         if weights and layer.name in weight_ranges:
    #             layer_range_info = weight_ranges[layer.name]
    #             new_weights = []
    #             for i, w in enumerate(weights):
    #                 if i == 0:
    #                     # Weight tensor (e.g. kernel)
    #                     range_info = layer_range_info['weight']
    #                 else:
    #                     # Bias tensor
    #                     range_info = layer_range_info['bias']
    #                 quantized = quantize_tensor_symmetric(w, range_info, num_bits=num_bits)
    #                 new_weights.append(quantized)
    #             layer.set_weights(new_weights)

    # This would quantize biases based on layer output
    for layer in model.layers:
        if hasattr(layer, "get_weights") and hasattr(layer, "set_weights"):
            weights = layer.get_weights()
            if weights and layer.name in wt_range_dict:
                layer_range_info = wt_range_dict[layer.name]
                new_weights = []
                for i, w in enumerate(weights):
                    if i == 0:
                        # Weight tensor (e.g. kernel)
                        try:
                            range_info = layer_range_info['weight']
                        except:
                            range_info = layer_range_info
                    else:
                        # Bias tensor
                        range_info = activation_range_dict[layer.name]
                    quantized = quantize_tensor_symmetric(w, range_info, num_bits=num_bits)
                    new_weights.append(quantized)
                layer.set_weights(new_weights)

    # evaluate new model
    if(mode=='eval'):
       acc, loss= model_evaluation_precise(model, batch_len=batch_len)

    acc = None
    loss=None

    return model, acc, loss


def quant_model(model, model_name, num_bits=8, design='hw', batch_len=157, force=0):
    # checks if for bw there is already a value in json, hanldes it as above
    dict = path_definition()
    BASE_PATH = dict['BASE_PATH']
    short_name = model_name[:-10]
    tmp_filepath = f"{BASE_PATH}/Docs_Reports/Quant/Metrics/{short_name}_metrics_quant_{design}.json"

    ask_message = 0
    save = 1
    try:
        with open(tmp_filepath, 'r') as f:
            metric_dict = json.load(f)
        if(isinstance(metric_dict[f'{num_bits}b']['accuracy'], float)):
            print(f'Read metrics from {tmp_filepath}')
            if(force==1):
                calculate = 1
                ask_message = 1
            else:
                calculate = 0
                ask_message = 0
                save = 0
    except:
        calculate = 1
        ask_message = 0

    if(calculate==1):
        qw_model, _, _ = quant_weights(model, model_name, num_bits=num_bits, mode='quant', design=design, batch_len=batch_len)
        qwa_model, acc, loss = quant_activations(qw_model, model_name, num_bits=num_bits, mode='eval', design=design, batch_len=batch_len)
    else:
        qwa_model = None

    # Shows message for the user to choose if they want to overwrite
    dict = path_definition()
    BASE_PATH = dict['BASE_PATH']
    save = 1
    if(ask_message==1 and BASE_PATH[:7]!='/kaggle'):
        while True:
            response = input("Do you want to overwrite previous data? (y/n): ").strip().lower()
            if response == 'y':
                save = 1
                break
            elif response == 'n':
                save = 0
                break
            else:
                print("Invalid input.")

    if(save==1):
        try:
            with open(tmp_filepath, 'r') as f:
                metric_dict = json.load(f)
        except:
            metric_dict = {}
        metric_dict[f'{num_bits}b'] = {'accuracy': float(acc), 'loss': float(loss)}
        parent_folder = os.path.dirname(tmp_filepath)
        os.makedirs(parent_folder, exist_ok=True)
        with open(tmp_filepath, 'w') as f:
            json.dump(metric_dict, f, indent=4)
    
    return qwa_model, acc, loss


def quant_bw_search(model, model_name, range):

    sw_metrics = {}
    hww_metrics = {}
    hwa_metrics = {}

    for i in range:
        print(f"Quantizing model to {i} bits...")
        _, acc, loss = quant_model(model, model_name, num_bits=i, design='sw', batch_len=1000, force=1)
        sw_metrics[f"{i}b"] = {"accuracy": float(acc), "loss": float(loss)}
        _, acc, loss = quant_model(model, model_name, num_bits=i, design='hww', batch_len=1000, force=1)
        hww_metrics[f"{i}b"] = {"accuracy": float(acc), "loss": float(loss)}
        _, acc, loss = quant_model(model, model_name, num_bits=i, design='hwa', batch_len=1000, force=1)
        hwa_metrics[f"{i}b"] = {"accuracy": float(acc), "loss": float(loss)}
        # add to json: sw, hww, hwa

    sorted_keys = sorted(sw_metrics.keys(), key=lambda x: int(x[:-1]))
    # plot
    # Extract x (labels), accuracies and losses
    x_labels = sorted_keys
    # Extract accuracy values
    sw_accuracy = [sw_metrics[k]["accuracy"] for k in sorted_keys]
    hww_accuracy = [hww_metrics[k]["accuracy"] for k in sorted_keys]
    hwa_accuracy = [hwa_metrics[k]["accuracy"] for k in sorted_keys]
    sw_loss = [sw_metrics[k]["loss"] for k in sorted_keys]
    hww_loss = [hww_metrics[k]["loss"] for k in sorted_keys]
    hwa_loss = [hwa_metrics[k]["loss"] for k in sorted_keys]

    # Plot accuracies
    plt.figure(figsize=(8, 4))
    plt.plot(sorted_keys, sw_accuracy, marker='o', label='SW Accuracy', color='blue')
    plt.plot(sorted_keys, hww_accuracy, marker='x', label='HWW Accuracy', color='orange')
    plt.plot(sorted_keys, hwa_accuracy, marker='s', label='HWA Accuracy', color='green')

    plt.title("Accuracy Comparison")
    plt.xlabel("Bit-width")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot losses

    plt.figure(figsize=(8, 4))
    plt.plot(sorted_keys, sw_loss, marker='o', label='SW Loss', color='blue')
    plt.plot(sorted_keys, hww_loss, marker='x', label='HWW Loss', color='orange')
    plt.plot(sorted_keys, hwa_loss, marker='s', label='HWA Loss', color='green')

    plt.title("Loss Comparison")
    plt.xlabel("Bit-width")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # print json
    print(json.dumps(sw_metrics, indent=4))
    print(json.dumps(hww_metrics, indent=4))
    print(json.dumps(hwa_metrics, indent=4))
    # plot the json


### Model transformation utilities

def quantize_tensor_symmetric(w, w_range, num_bits=8):
    qmin = -(2 ** (num_bits - 1) - 1)  # -127 for int8
    qmax = (2 ** (num_bits - 1) - 1)   # +127 for int8

    w_min = w_range["min"]
    w_max = w_range["max"]

    # Use symmetric range centered at 0
    max_abs = max(abs(w_min), abs(w_max))

    if max_abs == 0:
        return np.zeros_like(w)

    scale = max_abs / qmax  # ensure 0 maps to 0, and max_abs maps to ±127

    # Quantize
    q = np.round(w / scale)
    q = np.clip(q, qmin, qmax)

    # Dequantize
    w_dequant = q * scale

    return w_dequant


class SymmetricFakeQuantLayer(tf.keras.layers.Layer):
    def __init__(self, max_abs_val=6.0, num_bits=8, narrow_range=True, **kwargs):
        super().__init__(**kwargs)
        self.max_abs_val = max_abs_val
        self.min_val = -max_abs_val
        self.max_val = max_abs_val
        self.num_bits = num_bits
        self.narrow_range = narrow_range  # Set to True for signed int8 [-127, 127]

    def call(self, inputs):
        return tf.quantization.fake_quant_with_min_max_vars(
            inputs,
            min=self.min_val,
            max=self.max_val,
            num_bits=self.num_bits,
            narrow_range=self.narrow_range
        )


class SymmetricFakeQuantLayer_custom(tf.keras.layers.Layer):
    def __init__(self, max_abs_val=6.0, num_bits=8, narrow_range=True, **kwargs):
        super().__init__(**kwargs)
        self.max_abs_val = max_abs_val
        self.num_bits = num_bits
        self.narrow_range = narrow_range

    def call(self, inputs):
        # Symmetric int quant range
        if self.narrow_range:
            qmin = - (2 ** (self.num_bits - 1) - 1)
        else:
            qmin = - (2 ** (self.num_bits - 1))
        qmax = (2 ** (self.num_bits - 1)) - 1

        scale = self.max_abs_val / qmax

        x_clipped = tf.clip_by_value(inputs, -self.max_abs_val, self.max_abs_val)
        x_scaled = x_clipped / scale
        #x_rounded = tf.round(x_scaled)
        x_rounded = tf.floor(x_scaled)
        x_q = tf.clip_by_value(x_rounded, qmin, qmax)
        x_dequantized = x_q * scale

        return x_dequantized


class SymmetricFakeQuantLayer_custom_old(tf.keras.layers.Layer):
    def __init__(self, max_abs_val=6.0, num_bits=8, narrow_range=True, **kwargs):
        super().__init__(**kwargs)
        self.max_abs_val = max_abs_val
        self.num_bits = num_bits
        self.narrow_range = narrow_range

    def call(self, inputs):
        min_val = -self.max_abs_val
        max_val = self.max_abs_val

        qmin = 1.0 if self.narrow_range else 0.0
        qmax = 2.0**self.num_bits - 1.0

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = 0.0  # Symmetric quantization: zero-point is zero

        # Fake quantize: quantize then dequantize
        x_clipped = tf.clip_by_value(inputs, min_val, max_val)
        x_scaled = x_clipped / scale
        x_rounded = tf.round(x_scaled)
        x_q = tf.clip_by_value(x_rounded, qmin, qmax)
        x_dequantized = x_q * scale

        return x_dequantized


def clone_model_with_fake_quant(original_model, input_shape, range_dict, num_bits=8):
    new_model = Sequential()
    layer_mapping = []
    quant_layers_list = list(range_dict.keys())

    # Add input layer explicitly
    new_model.add(tf.keras.Input(shape=input_shape))

    quant_layer = 0
    for layer in original_model.layers:
        config = layer.get_config()
        cloned_layer = layer.__class__.from_config(config)
        # Insert fake quant after Conv2D or Dense
        if isinstance(cloned_layer, (Conv2D, Dense)):
            tmp_min = range_dict[quant_layers_list[quant_layer]]['min']
            tmp_max = range_dict[quant_layers_list[quant_layer]]['max']
            abs_max = abs(tmp_min) if abs(tmp_min)>tmp_max else tmp_max
            #new_model.add(FakeQuantLayer(min_val=tmp_min, max_val=tmp_max))
            # new_model.add(SymmetricFakeQuantLayer(max_abs_val=abs_max, num_bits=num_bits))
            new_model.add(SymmetricFakeQuantLayer_custom(max_abs_val=abs_max, num_bits=num_bits))
            quant_layer = quant_layer + 1
        # Clone layer from config
        new_model.add(cloned_layer)
        layer_mapping.append((layer, cloned_layer))

    # Build model by running dummy data through it
    dummy_input = tf.random.uniform((1, *input_shape))
    new_model(dummy_input)

    # Copy weights from original layers to cloned layers
    for orig_layer, cloned_layer in layer_mapping:
        if orig_layer.weights and cloned_layer.weights:
            try:
                cloned_layer.set_weights(orig_layer.get_weights())
            except ValueError as e:
                print(f"Skipping weights for layer {orig_layer.name} due to mismatch: {e}")

    new_model.build(input_shape=(None, *input_shape))  # Step 2

    dummy_input = tf.random.uniform((1, *input_shape))  # Step 3
    new_model(dummy_input)

    print("New model input shape:", new_model.input_shape)  # Step 4

    for orig_layer, cloned_layer in layer_mapping:
        try:
            cloned_layer.set_weights(orig_layer.get_weights())
        except Exception as e:
            print(f"Skipping weights for {orig_layer.name}: {e}")

    return new_model