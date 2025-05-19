import pandas as pd
import numpy as np   
import pickle

import io
from contextlib import redirect_stdout

from sklearn.linear_model import (
    Ridge,
    Lasso, 
    ElasticNet
)
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from pipeline.main import TrainingPipeline  



import dagshub
import mlflow
import logging
from dataclasses import dataclass
import os, sys
import mlflow.sklearn
from urllib.parse import urlparse
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from flask import (
    Flask,
    request,
    jsonify
)

app=Flask(__name__)

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
    

DATA_PATH=r"C:\Users\HP\Desktop\data\powerconsumption.csv"
ARTIFACT="artifact"
RAW_CSV="raw.csv"
TRAIN= "train.csv"
TEST="test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
DATA_TRANSFORMED_TRAIN_NUMPY_OBJECT = "train.npy"
DATA_TRANSFORMED_TEST_NUMPY_OBJECT = "test.npy"
TARGET_COLUMNS  = "PowerConsumption_Zone3"

TRAINED_MODEL_OBJECT = "model.pkl"


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

@dataclass
class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join(ARTIFACT, TRAINED_MODEL_OBJECT)

@dataclass
class MetricArtifact:
    Rmse: float
    mae: float
    mse: float
    r2score: float  


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: MetricArtifact
    test_metric_artifact: MetricArtifact 


def get_Regression_score(y_true,y_pred) -> MetricArtifact:
    try:
            
        model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        model_mae = mean_absolute_error(y_true, y_pred)
        model_mse=mean_squared_error(y_true,y_pred)
        model_r2score = r2_score(y_true,y_pred)

        Regression_metric =  MetricArtifact(Rmse=model_rmse, mae=model_mae, mse=model_mse,
                                                          r2score=model_r2score)
        return Regression_metric
    except Exception as e:
        logger.error(f"An Error Occured In : {e}")
        raise e



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
                self.remove_outliers_IQR(col, 
                                         train_df)
                self.remove_outliers_IQR(col, 
                                         test_df)

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

class ModelTrainerMethod:

    def __init__(self, data_trasformation_artifact : DataTransformationArtifact,
                 model_trainer_config : ModelTrainerConfig):
        
        try:
            self.data_trasformation_artifact = data_trasformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e
        
    def regression_tracking_mlfow(self, best_model, 
                                  regression_metrics):

        try:
            dagshub.init(
                repo_owner='Vishnuu011',
                repo_name="mlops_project",
                mlflow=True
            )

            url :str ="https://dagshub.com/Vishnuu011/mlops_project.mlflow"
            mlflow.set_registry_uri(url)
            experiment_name :str = "powerconsumptionmetrics"
            mlflow.set_experiment(experiment_name=experiment_name)

            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()
            ).scheme

            with mlflow.start_run():

                mlflow.log_metric("RMSE", regression_metrics.Rmse)
                mlflow.log_metric("MAE", regression_metrics.mae)
                mlflow.log_metric("MSE", regression_metrics.mse)
                mlflow.log_metric("R2SCOEE", regression_metrics.r2score)
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model_regression", registered_model_name="RegressionModel")
                else:
                    mlflow.sklearn.log_model(best_model, "model_classification")

        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e  

    def evaluate_models_regression(self, X_train, y_train,
                                   X_test,y_test,
                                   models,param) -> dict:

        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para=param[list(models.keys())[i]]

                gs = GridSearchCV(model,para,cv=3,verbose=2)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)

                #model.fit(X_train, y_train)  # Train model

                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)

                test_model_score = r2_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report

        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e    

    def train_model_regression(self,X_train,y_train,
                               x_test,y_test) -> ModelTrainerArtifact:

        try:
            models_reg = {
                "Random Forest": RandomForestRegressor(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "LGBMRegression": LGBMRegressor(),
                "ElasticNet": ElasticNet(),
                "XGBRegressor": XGBRegressor()
            }
            #regression hyperparametes
            params_reg = {
                "Ridge": {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
                },
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [20, 10],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 3],
                    'max_features': ['sqrt', 'log2'],  # Feature selection for regression
                },
                "Lasso": {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                    'selection': ['cyclic', 'random']
                },
                "ElasticNet": {
                    'alpha': [0.0001, 0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'selection': ['cyclic', 'random']
                },
                "XGBRegressor": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                },
                "LGBMRegression": {
                    'num_leaves': [31, 50, 70],
                    'max_depth': [1, 5, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [100, 200, 500]
                }
            }
            model_report_reg:dict=self.evaluate_models_regression(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                              models=models_reg,param=params_reg)
             
            print(model_report_reg)

            ## To get best model regression score from dict 
            best_model_score_reg = max(sorted(model_report_reg.values()))

            ## To get best model name from dict regression

            best_model_name_reg = list(model_report_reg.keys())[
                list(model_report_reg.values()).index(best_model_score_reg)
            ]
            print(best_model_name_reg)
            #regression 
            best_model_reg = models_reg[best_model_name_reg]
            y_train_pred=best_model_reg.predict(X_train)

            regression_train_metric=get_Regression_score(y_true=y_train,y_pred=y_train_pred)

             
            y_test_pred=best_model_reg.predict(x_test)
            regression_test_metric=get_Regression_score(y_true=y_test,y_pred=y_test_pred)

            self.regression_tracking_mlfow(best_model_reg,regression_test_metric)
            self.regression_tracking_mlfow(best_model_reg, regression_train_metric)

            #regrssion
            model_dir_path_r = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path_r,exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, best_model_reg)

            #model pusher regression
            save_object("final_model/model_regression.pkl",best_model_reg)
             

            ## Model Trainer Artifact
            model_trainer_artifact_r=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                        train_metric_artifact=regression_train_metric,
                                                        test_metric_artifact=regression_test_metric
                                                        )
            logging.info(f"Model trainer artifact: {model_trainer_artifact_r}")
            return model_trainer_artifact_r 
        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e

    def initiate_model_trainer(self)->tuple:


        try:
            train_file_path_regression = self.data_trasformation_artifact.transformed_train_file
            test_file_path_regression = self.data_trasformation_artifact.transformed_test_file

            train_arr_reg = load_numpy_array_data(train_file_path_regression)
            test_arr_reg = load_numpy_array_data(test_file_path_regression)
   

            #regression train and test split also drop both target columns 
            train_x, train_y, test_x, test_y = (
                train_arr_reg[:, :-1],
                train_arr_reg[:, -1],
                test_arr_reg[:, :-1],
                test_arr_reg[:, -1],
            )

             
            print("____________________________________________________________________________")

            print("======================Regresssion trainer started============================")
            print("_____________________________________________________________________________")

            model_trainer_artifact=self.train_model_regression(train_x, train_y, test_x, test_y )

            return model_trainer_artifact
        except Exception as e:
            raise e
        
class PredictionPipeline:

    def __init__(self):

        pass

    def predict(self, features):

        try:
            model_path=os.path.join("final_model", "model_regression.pkl")
            preprocessor_path=os.path.join("final_model", "preprocessor.pkl")
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            scaled_data = preprocessor.transform(features)
            preds = model.predict(scaled_data)
            return preds
        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e         


class CustomData:
    def __init__(  self,
        Temperature: float,
        Humidity: float,
        WindSpeed:float,
        GeneralDiffuseFlows: float,
        DiffuseFlows: float,
        ):

        self.Temperature = Temperature
        self.Humidity = Humidity
        self.WindSpeed = WindSpeed
        self.GeneralDiffuseFlows = GeneralDiffuseFlows
        self.DiffuseFlows = DiffuseFlows

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "Temperature": [self.Temperature],
                "Humidity": [self.Humidity],
                "WindSpeed": [self.WindSpeed],
                "GeneralDiffuseFlows": [self.GeneralDiffuseFlows],
                "DiffuseFlows": [self.DiffuseFlows]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e
        


@app.route("/predict", methods=['POST'])
def predict_datapoint():

    try:
        data=request.get_json()
        custom_data = CustomData(
            Temperature=data["Temperature"],
            Humidity=data["Humidity"],
            WindSpeed=data["WindSpeed"],
            GeneralDiffuseFlows=data["GeneralDiffuseFlows"],
            DiffuseFlows=data["DiffuseFlows"]
        )

        final_features = custom_data.get_data_as_data_frame()

        prediction_pipeline = PredictionPipeline()
        preds = prediction_pipeline.predict(final_features)
        return jsonify({"prediction": float(preds[0])})
    except Exception as e:
        logger.error(f"An Error Occured In : {e}")
        raise e
    


@app.route("/train", methods=['GET'])
def train_model():
    try:
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            pipeline = TrainingPipeline()
            artifact = pipeline.run_pipeline()
            print("\n✅ Training completed.")

        logs = buffer.getvalue()

        return jsonify({
            "status": "success",
            "message": "Training completed successfully",
            "model_path": getattr(artifact, 'model_path', "N/A"),
            "logs": logs
        }), 200

    except Exception as e:
        logger.error(f"An Error Occured In Training: {e}")
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True, port=5000)    