import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    steps = config.main.steps

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    original_cwd = hydra.utils.get_original_cwd()

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in steps:
            # This block runs the download step
            _ = mlflow.run(
                os.path.join(config["main"]["components_repository"], "get_data"),
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "raw_data.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw_file_as_downloaded"  # <--- THIS IS THE LINE
                },
            )

        # In main.py, inside the go() function...

        if "basic_cleaning" in steps:
            _ = mlflow.run(
                os.path.join(original_cwd, "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": config["basic_cleaning"]["input_artifact"],
                    "output_artifact": config["basic_cleaning"]["output_artifact"],
                    "output_type": config["basic_cleaning"]["output_type"],
                    "output_description": config["basic_cleaning"]["output_description"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

            # We run the command as a subprocess. If it fails, it will raise an exception
            subprocess.run(command, check=True)

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(original_cwd, "src", "data_check"),
                "main",
                parameters={
                    "csv": config["basic_cleaning"]["output_artifact"],
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    go()
