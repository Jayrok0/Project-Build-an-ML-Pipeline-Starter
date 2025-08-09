import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
import logging
import subprocess

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
        
        # We construct the full path for the output file
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
            "--ref", cleaned_data_path, # Using same data as reference
            "--kl_threshold", str(config["data_check"]["kl_threshold"]),
            "--min_price", str(config["etl"]["min_price"]),
            "--max_price", str(config["etl"]["max_price"]),
        ]
        
        env = os.environ.copy()
        subprocess.run(command, check=True, env=env)

    if "data_split" in active_steps:
        ##################
        # Implement here #
        ##################
        pass

    if "train_random_forest" in active_steps:
        # NOTE: we need to serialize the random forest configuration into JSON
        rf_config = os.path.abspath("rf_config.json")
        with open(rf_config, "w+") as fp:
            json.dump(dict(config["modeling"]["random_forest"].items()), fp)

        ##################
        # Implement here #
        ##################
        pass

if __name__ == "__main__":
    go()