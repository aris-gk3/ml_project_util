import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from .path import path_definition


### Implements all step of training and preprocessing

def load_preprocess():    
    dict = path_definition()
    PATH_DATASET = dict['PATH_DATASET']

    subfolders = [f.name for f in os.scandir(PATH_DATASET) if f.is_dir()]
    if(len(subfolders)==2):
        label_mode_str = 'binary'
    else:
        label_mode_str = 'categorical'

    train_dataset = image_dataset_from_directory(
        PATH_DATASET,
        image_size=(224, 224),
        batch_size=32,
        label_mode=label_mode_str,
        validation_split=0.2,  # 20% for validation
        subset='training',     # Use the 'training' subset
        seed=123
    )

    val_dataset = image_dataset_from_directory(
        PATH_DATASET,
        image_size=(224, 224),
        batch_size=32,
        label_mode=label_mode_str,
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


def test_split(percent=15, verbose=1):
    dict = path_definition()
    PATH_DATASET = dict['PATH_DATASET']
    PATH_TEST = dict['PATH_TEST']
    classes = [f for f in os.listdir(PATH_DATASET) if os.path.isdir(os.path.join(PATH_DATASET, f))]

    if verbose == 1:
        print_class_files(PATH_DATASET, PATH_TEST)

    for i in classes:
        move_random_files(f'{PATH_DATASET}/{i}', f'{PATH_TEST}/{i}', percent=percent)

    if verbose == 1:
        print_class_files(PATH_DATASET, PATH_TEST)


def undo_test_split(verbose=1):
    dict = path_definition()
    PATH_DATASET = dict['PATH_DATASET']
    PATH_TEST = dict['PATH_TEST']
    classes = [f for f in os.listdir(PATH_DATASET) if os.path.isdir(os.path.join(PATH_DATASET, f))]

    if verbose == 1:
        print_class_files(PATH_DATASET, PATH_TEST)

    for i in classes:
        move_back_files(f'{PATH_TEST}/{i}', f'{PATH_DATASET}/{i}')

    if verbose == 1:
        print_class_files(PATH_DATASET, PATH_TEST)


### Helper functions

def move_random_files(source_folder, destination_folder, percent=15):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # List all files in the source folder (excluding subdirectories)
    all_files = [f for f in os.listdir(source_folder)
                 if os.path.isfile(os.path.join(source_folder, f))]

    # Calculate number of files to move
    num_to_move = int(len(all_files) * percent / 100)

    # Randomly choose files to move
    files_to_move = random.sample(all_files, num_to_move)

    # Move files
    for filename in files_to_move:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved: {filename}")

    print(f"\nMoved {num_to_move} file(s) from '{source_folder}' to '{destination_folder}'.")


def move_back_files(source_folder, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # List all files in the source folder (excluding subdirectories)
    all_files = [f for f in os.listdir(source_folder)
                 if os.path.isfile(os.path.join(source_folder, f))]

    # Move files
    for filename in all_files:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)
        shutil.move(src_path, dst_path)
        print(f"Moved: {filename}")

    print(f"\nMoved all files from '{source_folder}' to '{destination_folder}'.")


def print_class_files(PATH_DATASET, PATH_TEST):
    # For validation & training set
    classes = [f for f in os.listdir(PATH_DATASET) if os.path.isdir(os.path.join(PATH_DATASET, f))]

    # Count number of images per folder
    num_files = []
    for i in classes:
        files = os.listdir(f'{PATH_DATASET}/{i}')  # list of all entries, which are all files in your case
        num_files.append(len(files))

    print('\n*** Training & Validation Classes are:\n')
    for i in range(len(classes)):
        print(f'{i}. {classes[i]} ({num_files[i]} images)')

    # For test set
    classes = [f for f in os.listdir(PATH_TEST) if os.path.isdir(os.path.join(PATH_TEST, f))]

    # Count number of images per folder
    num_files = []
    for i in classes:
        files = os.listdir(f'{PATH_TEST}/{i}')  # list of all entries, which are all files in your case
        num_files.append(len(files))

    print('\n*** Test Classes are:\n')
    for i in range(len(classes)):
        print(f'{i}. {classes[i]} ({num_files[i]} images)')