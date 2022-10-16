from housing.pipeline.pipeline import Pipeline
from housing.exception import HousingException
from housing.logger import logging
from housing.config.configration import Configration
from housing.component.data_transformation import DataTransformation
import os,sys

def main():
    try:
        
        config_path = os.path.join('config','config.yaml')

        pipeline = Pipeline(Configration(
            config_file_path=config_path
        ))
        pipeline.start()

        """data_validation_config = Configration().get_data_validation_config()
        print(data_validation_config)"""

        """data_transformation_config = Configration().get_data_transformation_config()
        print(data_transformation_config)"""

        """schema_file_path = r"D:\ineuron\ML_project\CS-winner\config\schema.yaml"
        file_path = r"D:\ineuron\ML_project\CS-winner\housing\artifact\data_ingestion\2022-09-29-01-30-53\ingested_dir\train\housing.csv"
        df = DataTransformation.load_data(file_path=file_path,schema_file_path=schema_file_path)
        print(df.columns)
        print(df.dtypes)"""
    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__=='__main__':
    main()