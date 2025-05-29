from ml_project_util.model_evaluation import model_evaluation_precise
from ml_project_util.path import path_definition
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
import json


_, _, pathRawData, _, pathSavedModels = path_definition()
filepath = f'{pathSavedModels}/CD3/CD3_P1_FT_continue_008_val0.0361.keras'
model = tf.keras.models.load_model(filepath)

print('Previously computed validation metrics\n')
with open(f'{pathRawData}/CD3_P1_FT_continue.json', 'r') as file:
    RawData = json.load(file)
print(f'Validation accuracy: {RawData['val_accuracy'][7]}')
print(f'Validation accuracy: {RawData['val_loss'][7]}')

model_evaluation_precise(model)