import pandas as pd
import numpy as np   
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass
import os, sys


logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')


logger = logging.getLogger()


logger.setLevel(logging.DEBUG)

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise  e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise e
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise  e
    

DATA_PATH="data\powerconsumption.csv"
ARTIFACT="artifact"
RAW_CSV="raw.csv"
TRAIN= "train.csv"
TEST="test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
DATA_TRANSFORMED_TRAIN_NUMPY_OBJECT = "train.npy"
DATA_TRANSFORMED_TEST_NUMPY_OBJECT = "test.npy"
TARGET_COLUMNS  = "PowerConsumption_Zone3"


@dataclass
class DataIngestionConfig:
    raw_data : str = os.path.join(ARTIFACT, RAW_CSV)
    train_path : str = os.path.join(ARTIFACT, TRAIN)
    test_path : str = os.path.join(ARTIFACT, TEST)


@dataclass
class DataIngestionArtifact:
    train_path : str
    test_path : str   

@dataclass
class DataTransformationConfig:
    transformed_object : str = os.path.join(ARTIFACT,
                                            PREPROCESSING_OBJECT_FILE_NAME)
    transformed_train_file : str = os.path.join(ARTIFACT,
                                                DATA_TRANSFORMED_TRAIN_NUMPY_OBJECT)
    transformed_test_file : str = os.path.join(ARTIFACT,
                                               DATA_TRANSFORMED_TEST_NUMPY_OBJECT)

@dataclass
class DataTransformationArtifact:
    transformed_object : str
    transformed_train_file : str
    transformed_test_file : str



class DataIngestionMethod:

    def __init__(self, data_path : str, 
                 data_ingestion_config : DataIngestionConfig):

        try:
            self.data_path = data_path
            self.data_ingestion_path = data_ingestion_config
        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e
        
    def export_data_from_local_folder(self) -> pd.DataFrame:

        try:
            logger.info("Reading csv dataset.....")
            data=pd.read_csv(self.data_path)
            logger.info(
                f"The Shape of Dataset : {data.shape}"
            )
            logger.info("Drop Some Columns......")
            data.drop(
                ['Datetime', 
                 'PowerConsumption_Zone1',
                 'PowerConsumption_Zone2',
                ],
                axis=1,
                inplace=True
            )
            logger.info(
                "Creating Folder For Storing Raw Data csv ....."
            )
            dir_name=os.path.dirname(self.data_ingestion_path.raw_data)
            os.makedirs(dir_name, exist_ok=True)
            logger.info(
                "After droping Some columns and created folder"
            )
            data.to_csv(
                self.data_ingestion_path.raw_data,
                index=False
            )
            logger.info(
                "Saved data in folder"
            )
            return data

        except Exception as e:
            logger.error(f"An Error Occured In : {e}")    
            raise e

    def raw_data_split_train_test(self):

        try:
            dataframe=self.export_data_from_local_folder()
            logger.info("train test split started .....")
            train, test = train_test_split(
                dataframe,
                test_size=0.2
            )
            dir_name=os.path.dirname(self.data_ingestion_path.train_path)
            os.makedirs(dir_name, exist_ok=True)
            train.to_csv(
                self.data_ingestion_path.train_path,
                index=False
            )
            dir2_name=os.path.dirname(self.data_ingestion_path.test_path)
            os.makedirs(dir2_name, exist_ok=True)
            test.to_csv(
                self.data_ingestion_path.test_path,
                index=False
            )
            logger.info("saved train and test")
            return train, test
        
        except Exception as e:
            logger.error(f"An Error Occured In : {e}")    
            raise e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:
            self.raw_data_split_train_test()

            artifact=DataIngestionArtifact(
                test_path=self.data_ingestion_path.test_path,
                train_path=self.data_ingestion_path.train_path
            )
            logger.info("Data Ingestion completed.....")
            return artifact
        except Exception as e:
            logger.error(f"An Error Occured In : {e}") 
            raise e   


class DataTransformationMethod:

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                  data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            logger.error(f"An error occurred in __init__: {e}")
            raise e

    @staticmethod
    def read_csv(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"An error occurred in read_csv: {e}")
            raise e

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            numeric_features = ['Temperature', 'Humidity', 'WindSpeed',
                            'GeneralDiffuseFlows', 'DiffuseFlows']
            remove_skew = ['GeneralDiffuseFlows', 'DiffuseFlows']
            other_features = list(set(numeric_features) - set(remove_skew))

            transformer = ColumnTransformer(transformers=[
                ("skew_and_scale", Pipeline(steps=[
                    ("cbrt", FunctionTransformer(func=np.cbrt, validate=True)),
                    ("scale", StandardScaler())
                ]), remove_skew),

                ("scale_only", StandardScaler(), other_features)
            ])
            return transformer

        except Exception as e:
            logger.error(f"An error occurred in get_data_transformer_object: {e}")
            raise e

    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 1.5 * IQR
            lower_limit = Q1 - 1.5 * IQR

            df.loc[df[col] > upper_limit, col] = upper_limit
            df.loc[df[col] < lower_limit, col] = lower_limit

            return df

        except Exception as e:
            logger.error(f"An error occurred in remove_outliers_IQR: {e}")
            raise e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Data Transformation Started...")

            logger.info("Reading train and test data...")
            train_df = DataTransformationMethod.read_csv(self.data_ingestion_artifact.train_path)
            test_df = DataTransformationMethod.read_csv(self.data_ingestion_artifact.test_path)

            logger.info("Removing outliers...")
            outlier_columns = ['Temperature', 'Humidity',
                               'GeneralDiffuseFlows', 'DiffuseFlows']

            for col in outlier_columns:
                self.remove_outliers_IQR(col, train_df)
                self.remove_outliers_IQR(col, test_df)

            logger.info("Splitting input and target features...")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMNS])
            target_feature_train_df = train_df[TARGET_COLUMNS]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMNS])
            target_feature_test_df = test_df[TARGET_COLUMNS]

            logger.info("Fitting and transforming data using preprocessor...")
            preprocessor = self.get_data_transformer_object()
            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)
            logger.info(f"preprocesser applyed {transformed_input_train_feature.shape, transformed_input_test_feature.shape}")

            logger.info("Concatenating transformed features with target...")
            train_arr = np.c_[
                transformed_input_train_feature,
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                transformed_input_test_feature,
                np.array(target_feature_test_df)
            ]
            logger.info(f"shape concatinate {train_arr.shape, test_arr.shape}")

            logger.info("Saving transformed arrays and preprocessor object...")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file, test_arr)
            save_object(self.data_transformation_config.transformed_object, preprocessor_object)
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            logger.info("Data Transformation Completed.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object=self.data_transformation_config.transformed_object,
                transformed_train_file=self.data_transformation_config.transformed_train_file,
                transformed_test_file=self.data_transformation_config.transformed_test_file
            )

            return data_transformation_artifact

        except Exception as e:
            logger.error(f"An error occurred in initiate_data_transformation: {e}")
            raise e    





