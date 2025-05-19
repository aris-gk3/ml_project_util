import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import preprocess_input
from utilpackage.path import path_definition

def load_preprocess():
    _, pathDataset, _, _, _ = path_definition()
    train_dataset = image_dataset_from_directory(
        pathDataset,
        image_size=(224, 224),
        batch_size=32,
        label_mode='binary',
        validation_split=0.2,  # 20% for validation
        subset='training',     # Use the 'training' subset
        seed=123
    )

    val_dataset = image_dataset_from_directory(
        pathDataset,
        image_size=(224, 224),
        batch_size=32,
        label_mode='binary',
        validation_split=0.2,  # 20% for validation
        subset='validation',   # Use the 'validation' subset
        seed=123
    )
    ## Preprocess & Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),  # 10% random rotation
        layers.RandomZoom(0.1),      # 10% zoom
        layers.RandomTranslation(0.1, 0.1),  # Random height and width shift
        layers.RandomBrightness(0.2)
    ])


    def augment_img(image, label):
        image = data_augmentation(image)  # Apply augmentations
        return image, label

    train_dataset = train_dataset.map(augment_img)

    def preprocess_img(image, label):
        image = preprocess_input(image)  # Apply VGG16-specific preprocessing
        return image, label

    train_dataset = train_dataset.map(preprocess_img)
    val_dataset = val_dataset.map(preprocess_img)

    return train_dataset, val_dataset