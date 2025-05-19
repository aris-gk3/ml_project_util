import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from dotenv import load_dotenv
from utilpackage.train import train, freeze_layers, unfreeze_head, unfreeze_block
from utilpackage.path import path_definition
from utilpackage.flatten_model import flatten_condtitional

TOY_MODEL = 1
TRAIN = 1
CONTINUE_TRAIN = 0
BLOCK_FINETUNE = 0

epochs = 1
lr = 1e-3
optimizer = 'Adam'
load_dotenv()  # loads variables from .env into os.environ
platform = os.getenv("PLATFORM")
name = 'Test_Model'

### For training
if TRAIN == 1:
    if TOY_MODEL == 1:
        model = tf.keras.Sequential()

        # Input Layer
        model.add(layers.Input(shape=(224, 224, 3)))

        # Add 18 Conv2D + BatchNorm + ReLU layers (Grouped to reduce size)
        for i in range(6):  # 6 blocks of 3 layers = 18 layers
            model.add(layers.Conv2D(32, (3, 3), padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())

        # Global average pooling
        model.add(layers.GlobalAveragePooling2D())  # Layer 19

        # Output layer for binary classification
        model.add(layers.Dense(1, activation='sigmoid'))  # Layer 20
    elif TOY_MODEL == 0:
        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in vgg_base.layers:
            layer.trainable = False
        model = models.Sequential([
            vgg_base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid') 
        ])


    train(model, epochs, lr, optimizer, name)

### For continuing (head or block) training
if CONTINUE_TRAIN == 1:
    if TOY_MODEL == 1:
        _, _, pathRawData, _, pathSavedModels = path_definition()
        filepath = f'{pathSavedModels}/CD3/CD3_P1_FT_continue_008_val0.0361.keras'
        model_saved = tf.keras.models.load_model(filepath)
        model = flatten_condtitional(model_saved, 'cd3_p1_ft_continue')
        freeze_layers(model, verbose=1)
        unfreeze_head(model, verbose=1)
        train(model, epochs, lr, optimizer, name, 'CD3_P1_FT_continue_008_val0.0361.keras')
    elif TOY_MODEL == 0:
        _, _, pathRawData, _, pathSavedModels = path_definition()
        filepath = f'{pathSavedModels}/Tes/Test_Model_001_val0.6640.keras'
        model_saved = tf.keras.models.load_model(filepath)
        # model = flatten_condtitional(model_saved, 'cd3_p1_ft_continue')
        freeze_layers(model, verbose=1)
        unfreeze_head(model, verbose=1)
        train(model, epochs, lr, optimizer, name, 'Test_Model_001_val0.6640.keras')

### For block fine tuning
if BLOCK_FINETUNE == 1:
    if TOY_MODEL == 1:
        _, _, pathRawData, _, pathSavedModels = path_definition()
        filepath = f'{pathSavedModels}/CD3/CD3_P1_FT_continue_008_val0.0361.keras'
        model_saved = tf.keras.models.load_model(filepath)
        model = flatten_condtitional(model_saved, 'cd3_p1_ft_continue')
        freeze_layers(model, verbose=1)
        unfreeze_head(model, verbose=1)
        unfreeze_block(model, verbose=1)
        train(model, epochs, lr, optimizer, name, 'CD3_P1_FT_continue_008_val0.0361.keras')
    elif TOY_MODEL == 0:
        _, _, pathRawData, _, pathSavedModels = path_definition()
        filepath = f'{pathSavedModels}/Tes/Test_Model_001_val0.6640.keras'
        model_saved = tf.keras.models.load_model(filepath)
        # model = flatten_condtitional(model_saved, 'cd3_p1_ft_continue')
        freeze_layers(model, verbose=1)
        unfreeze_head(model, verbose=1)
        unfreeze_block(model, verbose=1)
        train(model, epochs, lr, optimizer, name, 'Test_Model_001_val0.6640.keras')