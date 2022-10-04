from flask import g
from sklearn import preprocessing
from housing.exception import HousingException
from housing.logger import logging
from housing.entity.config_entity import DataTranformationConfig
from housing.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact,DataValidationArtifact
import os,sys
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from housing.util.util import read_yaml_file,save_object,save_numpy_array_data,load_data
from housing.constant import *

class FeatureGenerator(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True,
                total_rooms_index=3,
                population_index=5,
                households_index=6,
                total_bedrooms_index=4,
                columns=None) -> None:
        """
        FeatureGenerator Initialization
        add_bedrooms_per_room: bool
        total_rooms_index: int index number of total_rooms columns
        population_index:int index number of population columns
        households_index: int index number of households columns
        total_bedrooms_index: int index number of bedrooms columns
        """
        try:
            self.columns=columns
            if self.columns is not None:
                total_rooms_index = self.columns.index(COLUMN_TOTAL_ROOMS)
                population_index = self.columns.index(COLUMN_POPULATION)
                households_index = self.columns.index(COLUMN_HOUSEHOLDS)
                total_bedrooms_index = self.columns.index(COLUMN_TOTAL_BEDROOMS)

            self.add_bed_rooms_per_room = add_bedrooms_per_room
            self.total_rooms_index = total_rooms_index
            self.population_index = population_index
            self.total_bedrooms_index = total_bedrooms_index
            self.households_index = households_index
        except Exception as e:
            raise HousingException(e,sys) from e

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        try:
            room_per_household = X[:,self.total_rooms_index] / X[:,self.households_index]
            population_per_household = X[:,self.population_index] / X[:,self.households_index]

            if self.add_bed_rooms_per_room:
                bedrooms_per_room = X[:,self.total_bedrooms_index] / X[:,self.total_rooms_index]

                generated_feature = np.c_[X,room_per_household,population_per_household,bedrooms_per_room]
            else:
                generated_feature = np.c_[X,room_per_household,population_per_household]

            return generated_feature
        except Exception as e:
            raise HousingException(e,sys) from e


class DataTransformation:

    def __init__(self,data_transformation_config: DataTranformationConfig,
                    data_ingestion_artifact:DataIngestionArtifact,
                    data_validation_artifact:DataValidationArtifact) -> None:
        try:
            logging.info(f"{'='*20} Data Transformation log started. {'='*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise HousingException(e,sys) from e


    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)
            numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
            
            numerical_pipeline = Pipeline(steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('feature_generator',FeatureGenerator(
                    add_bedrooms_per_room=self.data_transformation_config.add_bedroom_per_room,
                    columns=numerical_columns)),
                ('scaler',StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessing = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_columns),
                ('categorical_pipeline',categorical_pipeline,categorical_columns)
            ])

            return preprocessing


        except Exception as e:
            raise HousingException(e,sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Obtaining training and test file paths.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and testing data as dataframes.")
            train_df = load_data(file_path=train_file_path,
                                schema_file_path=schema_file_path)

            test_df = load_data(file_path=test_file_path,
                                schema_file_path=schema_file_path)

            schema = read_yaml_file(schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]

            logging.info(f'Splitting data into input and target feature from training and testing dataframes.')
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[[target_column_name]]

            input_feature_test_df = test_df.drop([target_column_name],axis=1)
            target_feature_test_df = test_df[[target_column_name]]

            logging.info(f'Applying preprocessing object on trainingand testing datagrames.')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_name).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_name).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir,
                                                        train_file_name)

            transformed_test_file_path = os.path.join(transformed_test_dir,
                                                        test_file_name)
            logging.info(f"Saving transformed training and testign array.")
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path
            
            logging.info(f'saving preprocessing object.')
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(
                is_transformed=True,
                message="Data Transformation Successfull",
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_obj_file_path
            )


            logging.info(f"Data transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise HousingException(e,sys) from e



    


    