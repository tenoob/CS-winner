from housing.pipeline.pipeline import Pipeline
from housing.exception import HousingException
from housing.logger import logging
from housing.config.configration import Configration
import os,sys

def main():
    try:
        #pipeline = Pipeline()
        #pipeline.run_pipeline()

        """data_validation_config = Configration().get_data_validation_config()
        print(data_validation_config)"""

        data_transformation_config = Configration().get_data_transformation_config()
        print(data_transformation_config)
    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__=='__main__':
    main()