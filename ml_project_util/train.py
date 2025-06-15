import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam,AdamW, SGD # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from .load_preprocess import load_preprocess
from .path import path_definition
from .history import save_json
from .history import plot_history


### Function for quick training

def train(model, epochs, lr, optimizer, name, parent_name=None, is_binary=None, save_best=False, save_models=True, plot=False, augmentation_pipeline=None, early_stopping=False):
    train_dataset, val_dataset = load_preprocess(augmentation_pipeline=augmentation_pipeline)

    if optimizer == 'Adam':
        optimizer = Adam(learning_rate=lr)
    elif optimizer == 'AdamW':
        optimizer = AdamW(learning_rate=lr)
    elif optimizer == 'SGD':
        optimizer = SGD(learning_rate=lr)

    dict = path_definition()
    PATH_SAVEDMODELS = dict['PATH_SAVEDMODELS']

    if (is_binary == 1):
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    elif (is_binary == 0):
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        raise ValueError("Wrong flag for number of classes")

    folder_name = f"{PATH_SAVEDMODELS}/{name[:3]}"
    os.makedirs(folder_name, exist_ok=True)  # exist_ok=True avoids error if it already exists

    checkpoint_path = f"{PATH_SAVEDMODELS}/{name[:3]}/{name}_{{epoch:03d}}_val{{val_loss:.4f}}.keras"

    callbacks = []
    
    if (save_models):
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            save_freq='epoch',              # Save every epoch
            save_weights_only=False,
            save_best_only=save_best,           # Save every time, not just best
            monitor='val_loss',
            verbose=1
        )
        if (save_best):
            print('Will save only best models every epoch!')
        else:
            print('Will save all models every epoch!')
        callbacks.append(checkpoint_callback)

    if (early_stopping):
        early_stopping_callback = EarlyStopping(monitor='val_loss',
                                              patience=4)
        print('Will stop after 4 epochs of no improvement of validation loss!')
        callbacks.append(early_stopping_callback)
    else:
        print('No early stopping!')

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    if plot:
        plot_history(history)

    save_json(history, name, parent_name)

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