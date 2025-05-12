from huggingface_hub import snapshot_download
import os
import argparse
from pathlib import Path

def process_args():
    parser = argparse.ArgumentParser(
        description="Find out the relationship between tokens and experts"
    )
    parser.add_argument("--model", type=str, default="DeepSeek-V2")
    parser.add_argument("--dest", type=str, default="/root/filesystem/DeepSeek-V3")

    return parser.parse_args()



def main():
    args = process_args()
    dest_dir = Path(f"{args.dest}")
    
    n_cpus = os.cpu_count()               # logical cores on the machine
    print(f"Using {n_cpus} parallel workers")
    used_cpus = n_cpus - 4


    # downloads *only* the files you list in allow_patterns
    snapshot_download(
        repo_id=f"deepseek-ai/{args.model}",
        local_dir=dest_dir,
        resume_download=True,      # continue an interrupted run
        max_workers=used_cpus           # parallel downloads
    )


if __name__ == "__main__":
    main()