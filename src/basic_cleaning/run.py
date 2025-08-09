import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="basic_cleaning", project="nyc-airbnb")
    run.config.update(args)

    # Download input artifact
    logger.info("Downloading input artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    # Drop outliers
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Drop rows with missing values
    logger.info("Dropping rows with missing values")
    df = df.dropna(subset=['price', 'minimum_nights', 'host_name', 'latitude', 'longitude'])

    # Log the number of rows left
    logger.info(f"Number of rows left after cleaning: {df.shape[0]}")

    # Remove geographical outliers
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the cleaned data to a file
    logger.info("Saving cleaned data")
    df.to_csv("clean_sample.csv", index=False)

    # Upload the cleaned data to W&B as a new artifact
    logger.info("Creating and logging output artifact")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean the data and remove outliers",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact", type=str, help="Name of the input artifact"
    )

    parser.add_argument(
        "--output_artifact", type=str, help="Name of the output artifact"
    )

    parser.add_argument(
        "--output_type", type=str, help="Type of the output artifact"
    )

    parser.add_argument(
        "--output_description", type=str, help="Description of the output artifact"
    )

    parser.add_argument(
        "--min_price", type=float, help="Minimum price to consider"
    )

    parser.add_argument(
        "--max_price", type=float, help="Maximum price to consider"
    )

    args = parser.parse_args()
    go(args)