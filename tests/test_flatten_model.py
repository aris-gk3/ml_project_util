from ml_project_util.path import path_definition
from ml_project_util.model_evaluation import model_evaluation_precise
from ml_project_util.flatten_model import copy_all_weights
import tensorflow as tf
import json
from tensorflow.keras.applications import VGG16 # type: error
from tensorflow.keras import models, layers # type: error

# Load the VGG16 base model without the top (classifier), include weights
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = models.Sequential()

for layer in vgg_base.layers:
    model.add(layer)

model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

dict = path_definition()
pathBase = dict['BASE_PATH']
pathDataset = dict['PATH_DATASET']
pathTest = dict['PATH_TEST']
pathRawData = dict['PATH_RAWDATA']
pathJoinedData = dict['PATH_JOINEDDATA']
pathSavedModels = dict['PATH_SAVEDMODELS']

filepath = f'{pathSavedModels}/CD3/CD3_P1_FT_continue_008_val0.0361.keras'
model_old = tf.keras.models.load_model(filepath)

# Usage:
copy_all_weights(model_old, model)

# Evaluate metrics to verify correctness
model_evaluation_precise(model)
print('Previously computed validation metrics\n')
with open(f'{pathRawData}/CD3_P1_FT_continue.json', 'r') as file:
    RawData = json.load(file)
print(f'Validation accuracy: {RawData['val_accuracy'][7]}')
print(f'Validation accuracy: {RawData['val_loss'][7]}')