import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact
    logger.info("Downloading input artifact")
    # NOTE: this is the artifact we created in the "download" step
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    # Drop outliers
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Log the number of rows left
    logger.info(f"Number of rows after dropping price outliers: {len(df)}")

    # Drop rows with missing values
    df.dropna(inplace=True)
    # Log the number of rows left
    logger.info(f"Number of rows after dropping NaNs: {len(df)}")

    # Remove geographical outliers
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    logger.info(f"Number of rows after dropping geographical outliers: {len(df)}")


    # Save the cleaned data
    df.to_csv(args.output_artifact, index=False)

    # Upload the cleaned data to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A component that cleans the data and removes outliers.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The input artifact.",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The output artifact.",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type of the output artifact.",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A description for the output artifact.",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price.",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price.",
        required=True,
    )

    args = parser.parse_args()

    go(args)