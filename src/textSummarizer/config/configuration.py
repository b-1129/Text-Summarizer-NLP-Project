from textSummarizer.constants import *
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity import (DataIngestionConfig, 
                                   DataValidationConfig,
                                   DataTransformationConfig,
                                   ModelTrainerConfig,
                                   ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
            self,
            config_filepath= CONFIG_FILE_PATH,
            params_filepath= PARAMS_FILE_PATH):
        
            self.config=read_yaml(config_filepath)
            self.params=read_yaml(params_filepath)

            create_directories([self.config.artifacts_root])
        

    def get_data_ingestion_config(self) -> DataIngestionConfig:
            config= self.config.data_ingestion

            create_directories([config.root_dir])

            data_ingestion_config= DataIngestionConfig(
                root_dir= config.root_dir,
                source_URL= config.source_URL,
                local_data_file= config.local_data_file,
                unzip_dir= config.unzip_dir
        )
            return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config= DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config= DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.get('TrainingArguments', {})

        # Provide default values if necessary
        num_train_epochs = params.get('num_train_epochs', 3)
        warmup_steps = params.get('warmup_steps', 500)
        per_device_train_batch_size = params.get('per_device_train_batch_size', 1)
        weight_decay = params.get('weight_decay', 0.01)
        logging_steps = params.get('logging_steps', 10)
        evaluation_strategy = params.get('evaluation_strategy', 'steps')
        eval_steps = params.get('eval_steps', 500)
        save_steps = params.get('save_steps', 1e6)
        gradient_accumulation_steps = params.get('gradient_accumulation_steps', 16)

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(

            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            num_train_epochs=num_train_epochs,
            warmup_steps=warmup_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            save_steps=save_steps,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config= self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir= config.root_dir,
            data_path= config.data_path,
            model_path= config.model_path,
            tokenizer_path= config.tokenizer_path,
            metric_file_name= config.metric_file_name
        )

        return model_evaluation_config