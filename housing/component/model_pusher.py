import shutil
from housing.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
from housing.entity.config_entity import ModelPusherConfig
from housing.logger import logging
from housing.exception import HousingException
import os,sys

class ModelPusher:

    def __init__(self,model_pusher_config: ModelPusherConfig,
                 model_eval_artifact: ModelEvaluationArtifact) -> None:
        try:
            logging.info(f"{'>'*30} Model Pusher config {'<'*30}")
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
        except Exception as e:
            raise HousingException(e,sys) from e

    def export_model(self) -> ModelPusherArtifact:
        try:
            evaluated_model_file_path = self.model_eval_artifact.evaluated_model_path
            
            export_dir = self.model_pusher_config.export_dir_path
            
            model_file_name = os.path.basename(evaluated_model_file_path)

            export_model_file_path = os.path.join(
                export_dir,
                model_file_name
            )

            logging.info(f"Exported model file: [ {export_model_file_path} ]")
            os.makedirs(export_dir,exist_ok=True)

            shutil.copy(src=evaluated_model_file_path,dst= export_model_file_path)

            #db save code

            logging.info(f"Trained model: {evaluated_model_file_path} is copyed in export dir: [ {export_model_file_path} ]")

            model_pusher_artifact = ModelPusherArtifact(
                is_model_pushed=True,
                export_model_file_path=export_model_file_path
            )

            logging.info(f"Model pusher artifact: [ {model_pusher_artifact} ]")
            return model_pusher_artifact
        except Exception as e:
            raise HousingException(e,sys) from e


    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            return self.export_model()
        except Exception as e:
            raise HousingException(e,sys) from e


    def __del__(self):
        logging.info(f"{'>'*30} Model Pusher log Completed {'<'*30}")