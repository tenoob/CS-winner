from cmath import e
from dataclasses import is_dataclass
from housing.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from housing.entity.config_entity import ModelTrainerConfig
from housing.util.util import load_numpy_array_data, load_object, save_object
from housing.exception import HousingException
from housing.logger import logging
from housing.entity.model_factory import GridSearchBestModel, MetricInfoAritfact, ModelFactory, GridSearchBestModel , evaluate_regression_model
import os,sys
from typing import List

class HousingEstimatorModel:
    def __init__(self,preprocessing_object,trained_model_object) -> None:
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_objectL trained_model_object
        """
        try:
            self.preprocessing_object = preprocessing_object
            self.trained_model_object = trained_model_object
        except Exception as e:
            raise HousingException(e,sys) from e

    def predict(self,x):
        """
        Function accepts raw inputs and then transformed raw input using preprocessing object
        which gurantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        try:
            transformed_featuer = self.preprocessing_object.transform(x)
            return self.trained_model_object.predict(transformed_featuer)
        except Exception as e:
            raise HousingException(e,sys) from e

    def __repr__(self) -> str:
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) -> str:
        return f"P{type(self.trained_model_object).__name__}()"




class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact) -> None:
        try:
            logging.info(f"{'='*20} Model Trainer log started {'='*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise HousingException(e,sys) from e


    def initiate_model_trainer(self):
        try:
            logging.info(f"loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(file_path=transformed_train_file_path)

            logging.info(f"loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(file_path=transformed_test_file_path)

            logging.info("Splitting training and testing input and target feature")
            x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]

            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"initializing model factory class using above model config file: {model_config_file_path} ")
            model_factory = ModelFactory(model_config_path=model_config_file_path)

            base_acc = self.model_trainer_config.base_accuracy
            logging.info(f"Excepted Accuracy: {base_acc}")

            logging.info(f'Initiating operation model selection')
            best_model = model_factory.get_best_model(x=x_train,y=y_train,base_accuracy=base_acc)

            logging.info(f'best model found on training dataset: [{best_model}]')

            logging.info(f'Extracting trained model list')
            grid_searched_best_model_list: List[GridSearchBestModel] = model_factory.grid_searched_best_model_list

            model_list = [model.best_model for model in grid_searched_best_model_list]
            logging.info(f'Evaluation all trained model on training and testing dataset')
            metric_info:MetricInfoAritfact = evaluate_regression_model(model_list=model_list,
                                                                       x_train=x_train,
                                                                       y_train=y_train,
                                                                       x_test=x_test,
                                                                       y_test=y_test,
                                                                       base_accuracy=base_acc)

            logging.info(f"Best found model on both training and testing dataset")

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_object = metric_info.model_object

            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            housing_model = HousingEstimatorModel(
                                preprocessing_object=preprocessing_obj,
                                trained_model_object=model_object)

            logging.info(f'Saving model at path: {trained_model_file_path}')
            save_object(file_path=trained_model_file_path,obj=housing_model)

            model_trainer_artifact = ModelTrainerArtifact(
                is_trained=True,
                message="Model Trained Successfully",
                train_accuracy=metric_info.train_accuracy,
                test_accuracy= metric_info.test_accuracy,
                train_rmse= metric_info.train_rmse,
                test_rmse=metric_info.test_rmse,
                model_accuracy=metric_info.model_accuracy,
                trained_model_file_path=trained_model_file_path
            )
                
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise HousingException(e,sys) from e


    def __del__(self):
        logging.info(f"{'<<'*30} Model trainer log completed {'>>'*30}")