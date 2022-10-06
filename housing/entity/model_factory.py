import importlib
from statistics import mode
from tempfile import TemporaryFile
from tkinter import Menu
from tkinter.messagebox import NO
from typing import List
from housing.exception import HousingException
from housing.logger import logging
from housing.constant import *
from housing.util.util import read_yaml_file
from collections import namedtuple
import numpy as np
import os,sys
from sklearn.metrics import r2_score,mean_squared_error

InitializedModelDetail = namedtuple('InitializedModelDetail',
['model_serial_number','model','param_grid_search','model_name'])

GridSearchBestModel = namedtuple('GridSearchBestModel',
['model_serial_number','model','best_model','best_parameters','best_score'])

BestModel = namedtuple('BestModel',
['model_serial_number','model','best_model','best_parameters','best_score'])

MetricInfoAritfact = namedtuple('MetricInfoArtifact',
['model_name','model_object','train_rmse','test_rmse','train_accuracy','test_accuracy','model_accuracy','index_number'])

def evaluate_regression_model(model_list:list, x_train:np.array , y_train:np.array,
                              x_test:np.array,y_test:np.array,base_accuracy:float=0.6) -> MetricInfoAritfact:
    """
    Description:
    This function compare multiple regression model and return best model

    Params:
    model_list: list of model
    x_train: input feature of Training Dataset
    y_train: target feature of Training Dataset
    x_test: input feature of testing Dataset
    y_test: target feature of testing Dataset

    Return: 
    It returs a named tuple

    MetricInfoArtifact = namedTuple("MetricInfo",['model_name',
                                                  'model_object',
                                                  'train_rmse',
                                                  'test_rmse',
                                                  'train_accuracy',
                                                  'test_accuracy',
                                                  'model_accuracy',
                                                  'index_number'])                     
    """       
    try:
        index_number = 0
        metric_info_artifact = None

        for model in model_list:
            # getting model name based on model object
            model_name = str(model)
            logging.info(f"{'>>'*30} Started evaluating model: [{type(model).__name__}] {'<<'*30}")

            #getting prediction for training and testing dataset
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            #calculating r2 score for training and testing dataset
            train_acc = r2_score(y_train,y_train_pred)
            test_acc = r2_score(y_test,y_test_pred)

            #calculating mean squared error for training and testing dataset
            train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))

            #Calculating harmonic mean of train and test accuracy
            model_accuracy = (2*(train_acc*test_acc)/(train_acc+test_acc))
            diff_test_train_acc = abs(test_acc - train_acc)

            #logging all info
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f'Train Score \t\t Test Score \t\t Avg Score')
            logging.info(f'{train_acc} \t\t {test_acc} \t\t {model_accuracy}')

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f'Difference Test and Train accuracy: [{diff_test_train_acc}]')
            logging.info(f'Train root mean squared error: [{train_rmse}]')
            logging.info(f'Test root mean squared error: [{test_rmse}]')

            #if model acc is greater than base accuracy and train and test score is within certain threshold
            #we will accept that model as accepted model
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                base_accuracy = model_accuracy

                metric_info_artifact = MetricInfoAritfact(model_name=model_name,
                                                          model_object=model,
                                                          train_accuracy=train_acc,
                                                          test_accuracy=test_acc,
                                                          train_rmse=train_rmse,
                                                          test_rmse=test_rmse,
                                                          model_accuracy=model_accuracy,
                                                          index_number=index_number
                                                          )
                
                logging.info(f"Accepted model found: {metric_info_artifact}")

            index_number+=1

        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuarcy then base accuracy and train test score within certain threshold")

        return metric_info_artifact
    except Exception as e:
        raise HousingException(e,sys) from e

class ModelFactory:
    def __init__(self, model_config_path:str = None) -> None:
        try:
            self.config:dict = read_yaml_file(file_path=model_config_path)
            self.grid_search_cv_module :str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name:str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])

            self.model_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            self.initialization_model_list = None
            self.grid_searched_best_model_list = None
        except Exception as e:
            raise HousingException(e,sys) from e

    @staticmethod
    def update_property_of_class(instance_ref:object,property_data:dict):
        try:
            if not isinstance(property_data,dict):
                raise Exception('property_data parameter required to dictionary')
            print(property_data)

            for key,value in property_data.items():
                logging.info(f"Excecuting:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref,key,value)

            return instance_ref
        except Exception as e:
            raise HousingException(e,sys) from e


    @staticmethod
    def class_for_name(module_name:str,class_name:str):
        try:

            #loadign the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)

            #get the class, will raise AttribureError if class cannot be found
            logging.info(f'Executing command: from {module} import {class_name}')
            class_ref = getattr(module,class_name)
            return class_ref
        except Exception as e:
            raise HousingException(e,sys) from e

    def execute_grid_search_operation(self,initialized_model:InitializedModelDetail,
                                      input_feature,output_feaure) -> GridSearchBestModel:
        """
        execute_grid_search_operation(): function will perform parameter search operation and 
        it will return you the best optimistic model with best parameter:

        estimator: Model object
        param_grid: dict of parameter to perform search operation
        input_feature: all input features
        output_feauture: target feature

        return: Function will return GridSearchOperation object
        """
        try:
            #instantiating GridSearchCV class
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name)

            grid_search_cv = grid_search_cv_ref(estimator = initialized_model.model,
                                                param_grid = initialized_model.param_grid_search)

            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_property_data)

            message = f"{'>>'*30} Training {type(initialized_model.model).__name__} Started. {'<<'*30}"
            logging.info(message)

            grid_search_cv.fit(input_feature,output_feaure)

            message = f"{'>>'*30} Training {type(initialized_model.model).__name__} completed {'<<'*30}"

            grid_searched_best_model = GridSearchBestModel(model_serial_number=initialized_model.model_serial_number,
                                                            model=initialized_model.model,
                                                            best_model=grid_search_cv.best_estimator_,
                                                            best_parameters=grid_search_cv.best_params_,
                                                            best_score=grid_search_cv.best_score_)
            return grid_searched_best_model 

        except Exception as e:
            raise HousingException(e,sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        This function will return a list of model details.
        return Lists[ModelDetail]
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.model_initialization_config.keys():
                model_initialization_config = self.model_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY])
                model = model_obj_ref()

                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)
                
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"

                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                      model=model,
                                                                      param_grid_search=param_grid_search,
                                                                      model_name=model_name)

                initialized_model_list.append(model_initialization_config)

            self.initialization_model_list = initialized_model_list
            return self.initialization_model_list
        except Exception as e:
            raise HousingException(e,sys) from e

#289
    def initiate_best_parameter_search_for_initialized_model(self,initialized_model: InitializedModelDetail,
                                                             input_feature,output_feature) -> GridSearchBestModel:
        """
        Initiate_best_model_parameter_searhc(): function will perfrom parameter search operation
        and it will return you the best optimistic model with best parameter:
        estimator: Model object
        param_grid: dict of parameter to perform search operation
        input_feature: all inpt feature
        output_feature: dependent feature
        """
        try:
            return self.execute_grid_search_operation(
                initialized_model=initialized_model,
                input_feature=input_feature,
                output_feaure=output_feature
            )
        except Exception as e:
            raise HousingException(e,sys) from e
#//


    def initiate_best_parameter_search_for_initialized_models_list(self,initialized_model_list:List[InitializedModelDetail],
                                                             input_feature,output_feature) -> List[GridSearchBestModel]:  
        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(initialized_model=initialized_model_list,
                                                                                                     input_feature=input_feature,
                                                                                                     output_feature=output_feature)
                self.grid_searched_best_model_list.append(grid_searched_best_model)

            return self.grid_searched_best_model_list
        except Exception as e:
            raise HousingException(e,sys) from e

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        try:
            """
            This function return modelDetail
            """
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise HousingException(e,sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchBestModel],
                                                          base_accuracy=0.6) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found: {grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            
            if not best_model:
                raise Exception(f'None of Model has base accuracy: {base_accuracy}')
            logging.info(f"best model: {best_model}")
            return best_model
        except Exception as e:
            raise HousingException(e,sys) from e


    def get_best_model(self,x,y,base_accuracy=0.6) -> BestModel:
        try:
            logging.info(f"Started Initializing model from config file")

            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")

            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models_list(
                initialized_model_list=initialized_model_list,
                input_feature=x,
                output_feature=y
            )

            return ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list=grid_searched_best_model_list,
                base_accuracy=base_accuracy
            )
        except Exception as e:
            raise HousingException(e,sys) from e
