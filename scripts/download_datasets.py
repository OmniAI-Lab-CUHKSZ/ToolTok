"""Download all ToolTok training/evaluation datasets from ModelScope."""

from pathlib import Path

from modelscope.hub.snapshot_download import dataset_snapshot_download


DATASETS = {
    "rootsautomation/ScreenSpot": Path("data/screenspot"),
    "lmms-lab/ScreenSpot-Pro": Path("data/screenspot_pro"),
    "allenai/squad": Path("data/qa_squad"),
}


def main() -> None:
    for repo_id, local_dir in DATASETS.items():
        print(f"[ModelScope] {repo_id} -> {local_dir}", flush=True)
        dataset_snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            max_workers=8,
        )


if __name__ == "__main__":
    main()
