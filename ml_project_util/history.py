import json
import matplotlib.pyplot as plt
from .path import path_definition


### Utility functions for handling history object of training

# Path relative to Docs_Reports
def concatenate_json(relpath1, relpath2, output_filename):
    dict = path_definition()
    BASE_PATH = dict['BASE_PATH']
    PATH_JOINEDDATA = dict['PATH_JOINEDDATA']
    
    fullpath1 = f'{BASE_PATH}/Docs_Reports/{relpath1}'
    fullpath2 = f'{BASE_PATH}/Docs_Reports/{relpath2}'
    fullpath_out = f'{PATH_JOINEDDATA}/{output_filename}'
    with open(fullpath1) as f1, open(fullpath2) as f2:
        d1, d2 = json.load(f1), json.load(f2)

    result = {k: d1.get(k, []) + d2.get(k, []) for k in set(d1) | set(d2)}

    with open(fullpath_out, 'w') as out:
        json.dump(result, out, indent=4)

    print(f"Concatenation done. Saved to {fullpath_out}.")


# only name as argument
def plot_json(json_name, img_name, data_type='raw'):
    dict = path_definition()
    BASE_PATH = dict['BASE_PATH']
    PATH_RAWDATA = dict['PATH_RAWDATA']
    PATH_JOINEDDATA = dict['PATH_JOINEDDATA']

    if data_type=='raw':
        filepath = f'{PATH_RAWDATA}/{json_name}.json'
    elif data_type=='joined':
        filepath = f'{PATH_JOINEDDATA}/{json_name}.json'
    try:
        with open(filepath, 'r') as f:
            loaded_history = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        exit()

    # Get the list of metric names from the loaded history
    metric_names = list(loaded_history.keys())

    # Determine the number of subplots needed (one for each metric)
    num_metrics = len(metric_names)

    # Create the figure and subplots
    plt.figure(figsize=(15, 5 * num_metrics))

    # Plot Validation & Training Metrics
    for i, metric_name in enumerate(metric_names):

        # Check if there are validation metrics (e.g., val_loss, val_accuracy)
        if f'val_{metric_name}' in loaded_history:
            plt.subplot(num_metrics, 1, i + 1)
            metric_values = loaded_history[metric_name]
            val_metric_values = loaded_history[f'val_{metric_name}']
            epochs = range(1, len(metric_values) + 1)
            plt.plot(epochs, metric_values, 'b-', label=f'Training {metric_name}')
            plt.plot(epochs, val_metric_values, 'r-', label=f'Validation {metric_name}')
            plt.title(f'Training and Validation {metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.legend()
    
    plt.savefig(f'{BASE_PATH}/Docs_Reports/TrainingPlots/{img_name}.png', bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# only name as argument
def save_json(history, json_name, parent=None):
    dict = path_definition()
    PATH_RAWDATA = dict['PATH_RAWDATA']
    filepath = f"{PATH_RAWDATA}/{json_name}.json"

    # with open(filepath, 'w') as f:
    #     json.dump(history.history, f)

    data = {"Parent Model": parent}
    data.update(history.history)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


# only name as argument
def save_best_model_history(json_name, number, data_type='raw'):
    dict = path_definition()
    PATH_RAWDATA = dict['PATH_RAWDATA']
    PATH_JOINEDDATA = dict['PATH_JOINEDDATA']

    if data_type=='raw':
        filepath = f'{PATH_RAWDATA}/{json_name}.json'
    elif data_type=='joined':
        filepath = f'{PATH_JOINEDDATA}/{json_name}.json'
    
    with open(filepath, 'r') as f:
        history_dict = json.load(f)

    history_best = {}
    for key, value in history_dict.items():
        if isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
            # Truncate list of numbers
            history_best[key] = value[:number]
        else:
            # Leave other data types unchanged
            history_best[key] = value

    with open(filepath, 'w') as f:
        json.dump(history_best, f, indent=4)