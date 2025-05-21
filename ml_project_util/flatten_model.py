import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers

def copy_all_weights(old_model, new_model):
    # Functional layer -> many sequential layers
    
    new_layer_idx = 0  # index to track position in new_model.layers
    
    for old_layer in old_model.layers:
        if hasattr(old_layer, 'layers'):  # nested model layer (like vgg_base)
            # It's a nested model, loop through its layers
            for nested_layer in old_layer.layers:
                if type(nested_layer).__name__ == 'InputLayer':
                    continue
                new_layer = new_model.layers[new_layer_idx]
                if nested_layer.get_weights():
                    new_layer.set_weights(nested_layer.get_weights())
                new_layer_idx += 1
                
        else:
            # Normal layer, just copy weights
            new_layer = new_model.layers[new_layer_idx]
            if old_layer.get_weights():
                new_layer.set_weights(old_layer.get_weights())
            new_layer_idx += 1

def has_functional_layer(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and not isinstance(layer, tf.keras.Sequential):
            return True
    return False

def flatten_condtitional(model, name):
    if name.lower()[:3] == 'cd1':
        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model_new = models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
        for layer in vgg_base.layers:
            model_new.add(layer)
        model_new.add(layers.Flatten())
        model_new.add(layers.Dense(512, activation='relu'))
        model_new.add(layers.Dropout(0.5))
        model_new.add(layers.Dense(1, activation='sigmoid'))
    elif name.lower()[:3] == 'cd2':
        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model_new = models.Sequential()
        for layer in vgg_base.layers:
            model_new.add(layer)
        model_new.add(layers.GlobalAveragePooling2D())
        model_new.add(layers.Dense(128, activation='relu'))
        model_new.add(layers.Dropout(0.5))
        model_new.add(layers.Dense(1, activation='sigmoid'))
    elif name.lower()[:3] == 'cd3':
        vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model_new = models.Sequential()
        for layer in vgg_base.layers:
            model_new.add(layer)
        model_new.add(layers.GlobalAveragePooling2D())
        model_new.add(layers.Dense(256, activation='relu'))
        model_new.add(layers.BatchNormalization())
        model_new.add(layers.Dropout(0.5))
        model_new.add(layers.Dense(1, activation='sigmoid'))

    copy_all_weights(model, model_new)

    return model_new