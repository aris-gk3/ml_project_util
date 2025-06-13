### Find most used paths in all enviroments/machines

def path_definition(ds_rel_path=None):
    try: # Google Colab Definitions
        from google.colab import drive # type: ignore
        drive.mount('/content/drive')
        platform = 'Colab'
    except: 
        try: # Local Machines Definitions
            from dotenv import load_dotenv # type: ignore
            import os

            load_dotenv()  # loads variables from .env into os.environ

            platform = os.getenv("PLATFORM")
        except: # Kaggle Notebook Definitions
            platform = 'Kaggle'

    # Saves inside function & reuses ds_rel_path
    # Initialize saved config if it doesn't exist
    if not hasattr(path_definition, "config"):
        path_definition.config = {}
    # Set values if provided
    if ds_rel_path is not None:
        path_definition.config["ds_rel_path"] = ds_rel_path
        print(f"ds_rel_path set to: {ds_rel_path}")
    # Use stored values
    ds_rel_path = path_definition.config.get("ds_rel_path", "unknown")
            
    if platform == 'Kaggle':
        BASE_PATH = '/kaggle/working'
        PATH_DATASET = f"/kaggle/input/{ds_rel_path}/Train_val"
        PATH_TEST = f"/kaggle/input/{ds_rel_path}/Test"
        PATH_RAWDATA = f'{BASE_PATH}/Docs_Reports/RawTrainingData'
        PATH_JOINEDDATA = f'{BASE_PATH}/Docs_Reports/JoinedTrainingData'
        PATH_SAVEDMODELS = f'{BASE_PATH}/SavedModels'
    elif platform == 'Colab':
        BASE_PATH = f'/content/drive/MyDrive/Colab_Projects/{ds_rel_path}'
        PATH_DATASET = f'{BASE_PATH}/Dataset/Train_val'
        PATH_TEST = f'{BASE_PATH}/Dataset/Test'
        PATH_RAWDATA = f'{BASE_PATH}/Docs_Reports/RawTrainingData'
        PATH_JOINEDDATA = f'{BASE_PATH}/Docs_Reports/JoinedTrainingData'
        PATH_SAVEDMODELS = f'{BASE_PATH}/SavedModels'
    elif platform in ['Local', 'LocalRM', 'LocalOldLaptop']:
        # load from .env
        BASE_PATH = os.getenv("BASE_PATH")
        PATH_DATASET = os.getenv("PATH_DATASET")
        PATH_TEST = os.getenv("PATH_TEST")
        PATH_RAWDATA = os.getenv("PATH_RAWDATA")
        PATH_JOINEDDATA = os.getenv("PATH_JOINEDDATA")
        PATH_SAVEDMODELS = os.getenv("PATH_SAVEDMODELS")

    dict = {
        "BASE_PATH": BASE_PATH,
        "PATH_DATASET": PATH_DATASET,
        "PATH_TEST": PATH_TEST,
        "PATH_RAWDATA": PATH_RAWDATA,
        "PATH_JOINEDDATA": PATH_JOINEDDATA,
        "PATH_SAVEDMODELS": PATH_SAVEDMODELS
    }

    return dict
