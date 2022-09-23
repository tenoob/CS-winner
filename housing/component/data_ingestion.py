from threading import ExceptHookArgs
from token import EXACT_TOKEN_TYPES
from housing.entity.config_entity import DataIngestionConfig
from housing.exception import HousingException
from housing.logger import logging
import os,sys


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig) -> None:
        try:
            logging.info(f"{'='*20} Data Ingestion log started {'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HousingException(e,sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            pass
        except Exception as e:
            raise HousingException(e,sys) from e