import os
from datetime import datetime

Root_DIR = os.getcwd() # to get current working dir

CONFIG_DIR = 'config'  #the one in the base dir
CONFIG_FILE_NAME = 'config.yaml'
CONFIG_FILE_PATH =  os.path.join(Root_DIR,CONFIG_DIR,CONFIG_FILE_NAME)

CURRENT_TIME_STAMP = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

#Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = 'training_pipeline_config'
TRAINING_PIPELINE_ARTIFACT_KEY = 'artifact_dir'
TRAINING_PIPELINE_NAME_KEY = 'pipeline_name'