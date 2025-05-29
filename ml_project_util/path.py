### Find most used paths in all enviroments/machines

def path_definition():
    try: # Google Colab Definitions
        from google.colab import drive
        drive.mount('/content/drive')
        platform = 'Colab'
    except: 
        try: # Local Machines Definitions
            from dotenv import load_dotenv
            import os

            load_dotenv()  # loads variables from .env into os.environ

            platform = os.getenv("PLATFORM")
        except: # Kaggle Notebook Definitions
            platform = 'Kaggle'
            
    if platform == 'Kaggle':
        BASE_PATH = '/kaggle/working'
        PATH_DATASET = '/kaggle/input/catsdogsconv/KaggleCatsDogsConv'
        PATH_RAWDATA = f'{BASE_PATH}/RawTrainingData'
        PATH_JOINEDDATA = f'{BASE_PATH}/JoinedTrainingData'
        PATH_SAVEDMODELS = f'{BASE_PATH}/SavedModels'
    elif platform == 'Colab':
        BASE_PATH = '/content/drive/MyDrive/Colab_Projects/CatsDogs'
        PATH_DATASET = f'{BASE_PATH}/Dataset'
        PATH_RAWDATA = f'{BASE_PATH}/Docs_Reports/RawTrainingData'
        PATH_JOINEDDATA = f'{BASE_PATH}/Docs_Reports/JoinedTrainingData'
        PATH_SAVEDMODELS = f'{BASE_PATH}/SavedModels'
    elif platform in ['Local', 'LocalRM', 'LocalOldLaptop']:
        # load from .env
        BASE_PATH = os.getenv("BASE_PATH")
        PATH_DATASET = os.getenv("PATH_DATASET")
        PATH_RAWDATA = os.getenv("PATH_RAWDATA")
        PATH_JOINEDDATA = os.getenv("PATH_JOINEDDATA")
        PATH_SAVEDMODELS = os.getenv("PATH_SAVEDMODELS")

    return BASE_PATH, PATH_DATASET, PATH_RAWDATA, PATH_JOINEDDATA, PATH_SAVEDMODELS
