import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
import logging
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# List of pipeline steps
_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]

@hydra.main(config_name='config')
def go(config: DictConfig):
    # Set up W&B tracking
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Get the active steps and the original working directory
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps
    original_cwd = hydra.utils.get_original_cwd()

    # Download step (from remote repository)
    if "download" in active_steps:
        _ = mlflow.run(
            config["main"]["components_repository"] + "/get_data",
            "main",
            parameters={
                "sample": config["etl"]["sample"],
                "artifact_name": "raw_data.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw_file_as_downloaded"
            },
        )

    # Basic cleaning step (from local src directory, using subprocess for stability)
    if "basic_cleaning" in active_steps:
        logger.info("Running basic cleaning step")
        cleaned_data_path = os.path.join(original_cwd, config["basic_cleaning"]["output_artifact"])
        command = [
            "python", os.path.join(original_cwd, "src", "basic_cleaning", "run.py"),
            "--input_artifact", config["basic_cleaning"]["input_artifact"],
            "--output_artifact", config["basic_cleaning"]["output_artifact"],
            "--output_type", config["basic_cleaning"]["output_type"],
            "--output_description", config["basic_cleaning"]["output_description"],
            "--output_file", cleaned_data_path,
            "--min_price", str(config["etl"]["min_price"]),
            "--max_price", str(config["etl"]["max_price"]),
        ]
        env = os.environ.copy()
        subprocess.run(command, check=True, env=env)
    
    # Data check step (from local src directory, using subprocess for stability)
    if "data_check" in active_steps:
        logger.info("Running data check step")
        cleaned_data_path = os.path.join(original_cwd, config["basic_cleaning"]["output_artifact"])
        command = [
            "pytest", os.path.join(original_cwd, "src", "data_check"), "-vv",
            "--csv", cleaned_data_path,
            "--ref", cleaned_data_path,
            "--kl_threshold", str(config["data_check"]["kl_threshold"]),
            "--min_price", str(config["etl"]["min_price"]),
            "--max_price", str(config["etl"]["max_price"]),
        ]
        env = os.environ.copy()
        subprocess.run(command, check=True, env=env)

    # Data split step (implemented directly in main.py for stability)
    if "data_split" in active_steps:
        logger.info("Running data split step")
        run = wandb.init(job_type="data_split")
        
        cleaned_data_path = os.path.join(original_cwd, config["basic_cleaning"]["output_artifact"])
        df = pd.read_csv(cleaned_data_path)

        logger.info("Splitting data")
        trainval, test = train_test_split(
            df,
            test_size=config["modeling"]["test_size"],
            random_state=config["modeling"]["random_seed"],
            stratify=df[config["modeling"]["stratify_by"]]
        )

        # Save the splits to local files
        trainval_path = os.path.join(original_cwd, "trainval_data.csv")
        test_path = os.path.join(original_cwd, "test_data.csv")
        trainval.to_csv(trainval_path, index=False)
        test.to_csv(test_path, index=False)

        # Log the artifacts to W&B
        logger.info("Logging artifacts")
        trainval_artifact = wandb.Artifact("trainval_data.csv", type="split_data")
        trainval_artifact.add_file(trainval_path)
        run.log_artifact(trainval_artifact)

        test_artifact = wandb.Artifact("test_data.csv", type="split_data")
        test_artifact.add_file(test_path)
        run.log_artifact(test_artifact)

    # Train random forest step (from local src, using subprocess for stability)
    if "train_random_forest" in active_steps:
        logger.info("Running train_random_forest step")
        rf_config = os.path.abspath("rf_config.json")
        with open(rf_config, "w+") as fp:
            json.dump(dict(config["modeling"]["random_forest"].items()), fp)

        # We tell the training script to use the local file we just created
        trainval_path = os.path.join(original_cwd, "trainval_data.csv")

        command = [
            "python", os.path.join(original_cwd, "src", "train_random_forest", "run.py"),
            "--trainval_file", trainval_path, # Using local file
            "--val_size", str(config["modeling"]["val_size"]),
            "--random_seed", str(config["modeling"]["random_seed"]),
            "--stratify_by", config["modeling"]["stratify_by"],
            "--rf_config", rf_config,
            "--max_tfidf_features", str(config["modeling"]["max_tfidf_features"]),
            "--output_artifact", "random_forest_export",
        ]
        
        env = os.environ.copy()
        subprocess.run(command, check=True, env=env)

    if "test_regression_model" in active_steps:
        _ = mlflow.run(
            config["main"]["components_repository"] + "/test_regression_model",
            "main",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_dataset": "test_data.csv:latest"
            },
        )

if __name__ == "__main__":
    go()