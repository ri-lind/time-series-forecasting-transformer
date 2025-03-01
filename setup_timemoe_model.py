from huggingface_hub import snapshot_download
import json
import argparse


def download_hf_repo(repo_id, local_dir):
    """
    Download a snapshot of the Hugging Face repository specified by repo_id
    to the given local_dir.
    """
    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print("Download complete.")
    
    
def modify_config(time_MoE_directory: str):
    with open(f"{time_MoE_directory}/config.json", "r") as config:
        data = json.load(config)
    data["horizon_lengths"] = [1, 8, 32, 64, 128]
    with open(f"{time_MoE_directory}/config.json", "w") as config:
        json.dump(obj=data, fp=config)
    with open(f"{time_MoE_directory}/config.json", "r") as config:
        new_config_data = json.load(config)
    print(new_config_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--absolute_model_location', type=str, default="/content/time-moe")
    args = parser.parse_args()
    
    if args.absolute_model_location:
        time_moe_directory = args.absolute_model_location
        download_hf_repo("Maple728/TimeMoE-50M", time_moe_directory)
        modify_config(time_moe_directory)