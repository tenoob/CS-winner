from statistics import mode
from housing.constant import *
from housing.logger import logging
from housing.exception import HousingException
from housing.util.util import load_object, read_yaml_file, write_yaml_file
from housing.entity.config_entity import ModelEvaluationConfig
from housing.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
import os,sys


class ModelEvaluation:
    def __init__(self,model_evalation_config: ModelEvaluationConfig,
                 data_ingestion_config: DataIngestionArtifact,
                 data_validation_config: DataValidationArtifact,
                 model_trainer_config: ModelTrainerArtifact) -> None:
        try:
            logging.info(f"{'>'*30} Model Evaluation log started {'<'*30}")
            self.model_evaluation_config = model_evalation_config
            self.data_ingestion_config = data_ingestion_config
            self.data_validation_config = data_validation_config
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise HousingException(e,sys) from e

    def get_best_model(self):
        try:
            model = None
            #model_evaluation_file_path = self.model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY]
            model_evaluation_file_path = self.model_evaluation_config.model_evalaution_file_path

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path)
                return model
            
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise HousingException(e,sys) from e


    def update_evaluation_report(self,model_eval_artifact: ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evalaution_file_path

            model_eval_content = read_yaml_file(file_path=eval_file_path)

            model_eval_content = dict() if model_eval_content is None else model_eval_content

            current_deployed_model = None
            if BEST_MODEL_KEY in model_eval_content:
                current_deployed_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {current_deployed_model}")

            eval_result = {
                BEST_MODEL_KEY:{
                    MODEL_PATH_KEY: model_eval_artifact.evaluated_model_path
                }
            }

            if current_deployed_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: current_deployed_model}
                if HISTORY_KEY not in model_eval_content:
                    history =  {HISTORY_KEY:model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].updata(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result: {model_eval_content}")
            write_yaml_file(file_path=eval_file_path,data=model_eval_content)




        except Exception as e:
            raise HousingException(e,sys) from e