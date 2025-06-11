import argparse
import os
import pandas as pd
from datetime import datetime

from src.extract_features import extract_dataset_features
from src.train_model import train_and_save_model
from src.evaluate_model import evaluate_model
from src.utils.preprocessing import set_target_image_size

# --------------------------- Main Pipeline ---------------------------
def run_pipeline(args):
    # Setup output folders
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)

    # Set image target resize
    set_target_image_size(args.target_size)

    # Extract features
    print("[INFO] Extracting features...")
    df, num_images = extract_dataset_features(
         args.real_dir,
         args.fake_dir,
         args.features,
         max_images=args.total_images,
         cache_dir=args.cache_dir,
         selected_models=args.gen_models,
         metadata_path=args.metadata_csv
         )

    # Train and evaluate model
    print("[INFO] Training model...")
    model, scaler, X_test, y_test = train_and_save_model(df,
                                                        label_col="label",
                                                        model_type=args.model,
                                                        output_dir=os.path.join(args.output_dir, "models"),
                                                        test_size=args.test_size,
                                                        use_scaler=not args.no_scaling)

    print("[INFO] Evaluating model...")
    y_pred, y_true = evaluate_model(
        df, model_type=args.model, model_dir=os.path.join(args.output_dir, "models"), result_dir=os.path.join(args.output_dir, "results")
    )

    return num_images

# --------------------------- Argument Parsing ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full AI vs real image classification pipeline.")

    parser.add_argument("--real_dir", type=str, default="data/real", help="Directory of real images")
    parser.add_argument("--fake_dir", type=str, default="data/fake", help="Directory of fake images")
    parser.add_argument("--features", nargs="+", required=True, help="List of feature names to extract")
    parser.add_argument("--model", required=True, help="Model to train")
    parser.add_argument("--total_images", type=int, default=None, help="Total number of images to use from each class")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of dataset to use for testing")
    parser.add_argument("--target_size", type=int, default=256, help="Target size for resized images")
    parser.add_argument("--cache_dir", type=str, default="src/cache", help="Cache directory for features")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Folder to save models/results")
    parser.add_argument("--no_scaling", action="store_true", default=True, help="Skip feature scaling before training")
    parser.add_argument("--gen_models", nargs="*", default=None, help="List of generative models to filter fake images")
    parser.add_argument("--metadata_csv", type=str, default="data/fake/fake_metadata.csv", help="Path to fake image metadata")

    args = parser.parse_args()
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # Make per-run subdirectory and update output_dir
    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_subdir = os.path.join(args.output_dir, run_name)
    args.output_dir = output_subdir

    images_used = run_pipeline(args)

    # Store config + metrics
    log_entry = {
        "run_time": datetime.now().isoformat(),
        "log_name": run_name,
        "model": args.model,
        "features": args.features,
        "total_images": images_used,
        "test_size": args.test_size,
        "target_size": args.target_size,
        "output_dir": output_subdir,
        "log_scaling": not args.no_scaling,
        "selected_gen_models": args.gen_models,
        }

    log_df = pd.DataFrame([log_entry])
    log_file = os.path.join(args.output_dir, "..", "run_log.csv")
    log_df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
