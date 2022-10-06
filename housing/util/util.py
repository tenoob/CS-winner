from dbm import dumb
from genericpath import exists
import numpy as np
from housing.exception import HousingException
import os,sys,yaml
import dill
import pandas as pd
from housing.constant import *

def read_yaml_file(file_path:str) -> dict:
    """
    Reads a YAML file and returns the contents as a dict
    file_path: str
    """
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HousingException(e,sys) from e

def save_numpy_array_data(file_path:str , array:np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise HousingException(e,sys) from e

def load_numpy_array_data(file_path:str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise HousingException(e,sys) from e


def save_object(file_path:str , obj):
    """
    file_path: str
    obj: any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise HousingException(e,sys) from e


def load_object(file_path:str):
    """
    file_path: str
    """
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise HousingException(e,sys) from e


def load_data(file_path: str , schema_file_path:str) -> pd.DataFrame:
        try:
            dataset_schema = read_yaml_file(schema_file_path)

            schema = dataset_schema[DATASET_SCHEMA_COLUMNS_KEY]

            dataframe = pd.read_csv(file_path)

            error_message = ""

            for column in dataframe.columns:
                if column in list(schema.keys()):
                    dataframe[column].astype(schema[column])
                else:
                    error_message = f"{error_message} \nColumn: [{column}] is not in the schema."
            if len(error_message) > 0:
                raise Exception(error_message)

            return dataframe

        except Exception as e:
            raise HousingException(e,sys) from e


def get_sample_model_config_yaml_file(export_dir:str):
    try:
        model_config = {
            GRID_SEARCH_KEY:{
                MODULE_KEY: 'sklearn.model_selection',
                CLASS_KEY: 'GridSearchCV',
                PARAM_KEY: {
                    'cv':5,
                    'verbose':2
                }
            },
            MODEL_SELECTION_KEY: {
                'module_0':{
                    MODULE_KEY:'module_of_model',
                    CLASS_KEY: 'ModelClassName',
                    PARAM_KEY:
                    {
                        'param_name1': 'value1',
                        'param_name2':'value2',
                    },
                    SEARCH_PARAM_GRID_KEY:
                    {
                        'param_name':['param_value1','param_value2']
                    }
                },
            }
        }

        os.makedirs(export_dir,exist_ok=True)
        export_file_path = os.path.join(export_dir,"model.yaml")
        with open(export_file_path,'w') as file:
            yaml.dump(model_config,file)
        return export_file_path
    except Exception as e:
        raise HousingException(e,sys) from e



