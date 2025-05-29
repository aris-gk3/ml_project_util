from .path import path_definition
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import preprocess_input

### Functions for evaluating the model on the training dataset

def model_evaluation(model):
    _, PATH_DATASET, _, _, _ = path_definition()
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

def model_evaluation_precise(model, batch_len=157):
    _, pathDataset, _, _, _ = path_definition()
    val_dataset = image_dataset_from_directory(
        pathDataset,
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

    acc_metric = tf.keras.metrics.BinaryAccuracy()
    loss_metric = tf.keras.metrics.Mean()  # To average the loss over batches
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    batch_no = 0
    for batch in val_dataset:
        if batch_no < batch_len:
            print(f'Batch Number: {batch_no}')
            batch_no = batch_no + 1

            images, labels = batch
            preds = model(images, training=False)
            acc_metric.update_state(labels, preds)

            loss = loss_fn(labels, preds)
            loss_metric.update_state(loss)

    final_acc = acc_metric.result().numpy()
    final_loss = loss_metric.result().numpy()

    print(f"Precise val accuracy: {final_acc:.5f}")
    print(f"Precise val loss: {final_loss:.5f}")