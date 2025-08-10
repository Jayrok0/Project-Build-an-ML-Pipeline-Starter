import argparse
import logging
import pandas as pd
import wandb
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading input artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    logger.info("Dropping outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    logger.info("Dropping rows with missing values")
    df.dropna(inplace=True)
    logger.info(f"Number of rows left after cleaning: {df.shape[0]}")

    logger.info(f"Saving cleaned data to {args.output_file}")
    df.to_csv(args.output_file, index=False)

    logger.info("Creating and logging output artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_file)
    run.log_artifact(artifact)

    logger.info(f"Number of rows after cleaning: {df.shape[0]}")

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info(f"Saving cleaned data to {args.output_file}")
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to clean the Airbnb dataset.")
    parser.add_argument("--input_artifact", type=str, help="W&B artifact to use as input", required=True)
    parser.add_argument("--output_artifact", type=str, help="Name for the output artifact", required=True)
    parser.add_argument("--output_type", type=str, help="Type of the output artifact", required=True)
    parser.add_argument("--output_description", type=str, help="Description for the output artifact", required=True)
    parser.add_argument("--output_file", type=str, help="Path to the output file", required=True) # New argument
    parser.add_argument("--min_price", type=float, help="Minimum price", required=True)
    parser.add_argument("--max_price", type=float, help="Maximum price", required=True)
    
    args = parser.parse_args()
    go(args)