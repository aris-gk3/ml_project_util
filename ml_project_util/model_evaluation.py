import os
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from .path import path_definition

### Functions for evaluating the model on the training dataset

def model_evaluation(model):
    dict = path_definition()
    PATH_DATASET = dict['PATH_DATASET']

    val_dataset = image_dataset_from_directory(
        PATH_DATASET,
        image_size=(224, 224),
        batch_size=32,
        label_mode='binary',
        validation_split=0.2,  # 20% for validation
        subset='validation',   # Use the 'validation' subset
        seed=123
    )
    # Apply VGG-16 preprocessing
    def preprocess_img(image, label):
        image = preprocess_input(image)  # Apply VGG16-specific preprocessing
        return image, label
    
    val_dataset = val_dataset.map(preprocess_img)

    loss, accuracy = model.evaluate(val_dataset)

    print(f'Validation loss is: {loss}')
    print(f'Validation accuracy is: {accuracy}')

def model_evaluation_precise(model, batch_len=157, mode='test'):
    dict = path_definition()
    PATH_DATASET = dict['PATH_DATASET']
    PATH_TEST = dict['PATH_TEST']

    if (model.loss == 'categorical_crossentropy'):
        label_mode_str = 'categorical'
    elif (model.loss == 'binary_crossentropy'):
        label_mode_str = 'binary'
    else:
        raise ValueError("Loss function of model is not recognized!")

    if (mode=='val'):
        dataset = image_dataset_from_directory(
            PATH_DATASET,
            image_size=(224, 224),
            batch_size=32,
            label_mode=label_mode_str,
            validation_split=0.2,  # 20% for validation
            subset='validation',   # Use the 'validation' subset
            seed=123
        )
    elif (mode=='test'):
        dataset = image_dataset_from_directory(
        PATH_TEST,
        image_size=(224, 224),
        batch_size=32,
        label_mode=label_mode_str,
        shuffle=False  # usually no shuffle on test set
    )

    # Apply VGG-16 preprocessing
    def preprocess_img(image, label):
        image = preprocess_input(image)  # Apply VGG16-specific preprocessing
        return image, label

    dataset = dataset.map(preprocess_img)

################
    # loss_metric = tf.keras.metrics.Mean()  # To average the loss over batches
    if (model.loss == 'categorical_crossentropy'):
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        acc_metric = tf.keras.metrics.CategoricalAccuracy()
    elif (model.loss == 'binary_crossentropy'):
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        acc_metric = tf.keras.metrics.BinaryAccuracy()
    else:
        raise ValueError("Loss function of model is not recognized!")
##############
    print('Start evaluating batches')
    batch_no = 0
    total_loss = 0.0
    total_samples = 0
    for batch in dataset:
        if batch_no < batch_len:
            print(f'\rBatch Number: {batch_no}', end='', flush=True)
            batch_no = batch_no + 1

            images, labels = batch
            preds = model(images, training=False)
            acc_metric.update_state(labels, preds)

            # Compute batch loss and accumulate sample-wise
            batch_loss = loss_fn(labels, preds).numpy()
            batch_size = labels.shape[0]
            total_loss += batch_loss * batch_size
            total_samples += batch_size

    final_acc = acc_metric.result().numpy()
    # final_loss = loss_metric.result().numpy()
    # final_loss = loss_metric.result().numpy() / (batch_len * 32)  # total samples = batches * batch_size
    final_loss = total_loss / total_samples  # average per sample

    print(f"\nPrecise {mode} accuracy: {final_acc:.5f}")
    print(f"Precise {mode} loss: {final_loss:.5f}")

    return final_acc, final_loss