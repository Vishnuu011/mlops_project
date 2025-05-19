from app import (
    DataIngestionConfig,
    DataIngestionArtifact,
    DataTransformationConfig,
    DataTransformationArtifact,
    DataIngestionMethod,
    DataTransformationMethod
)
from app import *


class TrainingPipeline:

    def __init__(self):
        
        try:
            self.data_path=DATA_PATH
            self.data_ingestion_config=DataIngestionConfig()
            self.data_trasformation_config=DataTransformationConfig()
            self.model_trainer_config=ModelTrainerConfig()

        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e
        
    def strat_data_ingestion(self):

        try:
            ingestion_obj=DataIngestionMethod(
                data_path=self.data_path,
                data_ingestion_config=self.data_ingestion_config
            )
            ingestion_artifact=ingestion_obj.initiate_data_ingestion()
            return ingestion_artifact
        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e 

    def strat_data_Transformation(self, data_ingestion_artifact):

        try:
            Transformation_obj=DataTransformationMethod(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_trasformation_config
            )
            trasformation_artifact=Transformation_obj.initiate_data_transformation()
            return trasformation_artifact
        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e   

    def strat_model_trainer(self, data_trasformation_artifact):

        try:
            model_trainer_obj=ModelTrainerMethod(
                data_trasformation_artifact=data_trasformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifact=model_trainer_obj.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e           

    def run_pipeline(self):

        try:
            data_ingestion_artifact=self.strat_data_ingestion()
            data_transformation_artifact=self.strat_data_Transformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            model_artifact =self.strat_model_trainer(
                data_trasformation_artifact=data_transformation_artifact
            )
            return model_artifact
        except Exception as e:
            logger.error(f"An Error Occured In : {e}")
            raise e         
        