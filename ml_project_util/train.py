import tensorflow as tf
from tensorflow.keras.optimizers import Adam,AdamW, SGD # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from ml_project_util.load_preprocess import load_preprocess
from ml_project_util.path import path_definition
from ml_project_util.history import save_json


### Function for quick training

def train(model, epochs, lr, optimizer, name, parent_name=None):
    train_dataset, val_dataset = load_preprocess()

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'AdamW':
        optimizer = AdamW(learning_rate=lr)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=lr)

    dict = path_definition()
    PATH_SAVEDMODELS = dict['PATH_SAVEDMODELS']
    # if(binary===1):
    #     model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # else:
    #     model.compile(optimizer=optimizer, loss='categorical_crossentropy ', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy ', metrics=['accuracy'])
    checkpoint_path = f"{PATH_SAVEDMODELS}/{name[:3]}/{name}_{{epoch:03d}}_val{{val_loss:.4f}}.keras"

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq='epoch',              # Save every epoch
        save_weights_only=False,
        save_best_only=False,           # Save every time, not just best
        monitor='val_loss',
        verbose=1
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[checkpoint_callback]
    )

    save_json(history, name, parent_name)
    # filepath = f"{PATH_RAWDATA}/{name}.json"
    # with open(filepath, 'w') as f:
    #     json.dump(history.history, f)

def freeze_layers(model, verbose=0):
    for layer in model.layers[:]:
        layer.trainable = False
    if(verbose):
        print_model_layers(model)
        print('\n')

def unfreeze_head(model, verbose=0):
    for layer in model.layers[18:]:
        layer.trainable = True   

    if(verbose):
        print_model_layers(model)
        print('\n')

def unfreeze_block(model, verbose=0):
    for layer in model.layers:
        if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
            layer.trainable = True

    if(verbose):
        print_model_layers(model)
        print('\n')


### Training Visualization

def print_model_layers(model, indent=0):
    for layer in model.layers:
        print(" " * indent + f"- {layer.name} ({layer.__class__.__name__}), Trainable: {layer.trainable}")
        # If this layer has sublayers (like Functional or Sequential models)
        if hasattr(layer, 'layers'):
            print_model_layers(layer, indent + 2)