import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip optimizer state from a PyTorch/MMCV checkpoint.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    slim = {
        "meta": checkpoint.get("meta", {}),
        "state_dict": checkpoint["state_dict"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, args.output)

    print(f"Wrote slim checkpoint: {args.output}")


if __name__ == "__main__":
    main()
