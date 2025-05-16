from .app import (
    DataIngestionConfig,
    DataIngestionArtifact,
    DataTransformationConfig,
    DataTransformationArtifact,
    DataIngestionMethod,
    DataTransformationMethod
)
from .app import *

if __name__ == '__main__':

    obj=DataIngestionMethod(
        data_path=DATA_PATH,
        data_ingestion_config=DataIngestionConfig()
    )
    ingestion_artifact=obj.initiate_data_ingestion()
    print(ingestion_artifact)
    transformed_obj=DataTransformationMethod(
        data_ingestion_artifact=ingestion_artifact,
        data_transformation_config=DataTransformationConfig()
    )
    transformed_artifact=transformed_obj.initiate_data_transformation()
    print(transformed_artifact)