# MLProjectUtil

A small package hosted in Github for deploying quickly, reusable code in my project across various platforms.

Main functions are:

1. Transfer Learning Training
2. Data Handling (Training History & Metadata) in Json files
3. Plotting

## Requirements

For local execution: .env file with BASE_PATH, PATH_DATASET, PATH_RAWDATA, PATH_JOINEDDATA, PATH_SAVEDMODELS

1. Python = 3.12.9
2. tensorflow = 2.19.0
3. matplotlib = 3.10.0
4. dotenv = 0.9.9

To install on cloud notebooks ```!pip install git+https://github.com/aris-gk3/ml_project_util.git ```

## Testing

Manual execution of test*.py files & check of results.

## Example Code

Training:

```python
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = models.Sequential()
for layer in vgg_base.layers:
    model.add(layer)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation='sigmoid'))

epochs = 3
lr = 1e-3
optimizer = 'Adam'
load_dotenv()  # loads variables from .env into os.environ
platform = os.getenv("PLATFORM")
name = 'CD4_P1'

freeze_layers(model, verbose=1)
unfreeze_head(model, verbose=1)
train(model, epochs, lr, optimizer, name)
```

Continue training:

```python
# Load model
_, _, pathRawData, _, pathSavedModels = path_definition()
filepath = f'{pathSavedModels}/CD3/CD3_P1_FT_continue_008_val0.0361.keras'
model = tf.keras.models.load_model(filepath)
# Some saved models need transformation
flatten_condtitional(model, 'cd3_p1_ft_continue')
train(model, epochs, lr, optimizer, name)
```

Fine tune last block:

```python
# Load model
_, _, pathRawData, _, pathSavedModels = path_definition()
filepath = f'{pathSavedModels}/CD3/CD3_P1_FT_continue_008_val0.0361.keras'
model = tf.keras.models.load_model(filepath)
# Some saved models need transformation
flatten_condtitional(model, 'cd3_p1_ft_continue')
unfreeze_block(model, verbose=1)
train(model, epochs, lr, optimizer, name)
```