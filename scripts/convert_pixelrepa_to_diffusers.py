import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

try:
    from src.diffusers.models.transformers.transformer_pixelrepa import PixelREPATransformer2DModel
    from src.diffusers.pipelines.pixelrepa.pipeline_pixelrepa import PixelREPAPipeline
    from src.diffusers.schedulers.scheduling_pixelrepa import PixelREPAScheduler
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.diffusers.models.transformers.transformer_pixelrepa import PixelREPATransformer2DModel
    from src.diffusers.pipelines.pixelrepa.pipeline_pixelrepa import PixelREPAPipeline
    from src.diffusers.schedulers.scheduling_pixelrepa import PixelREPAScheduler


def get_args():
    parser = argparse.ArgumentParser(description="Convert legacy PixelREPA checkpoint to Diffusers format.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to PixelREPA checkpoint (.pth).")
    parser.add_argument("--output_dir", type=str, required=True, help="Output Diffusers model directory.")
    parser.add_argument(
        "--weights",
        type=str,
        default="ema1",
        choices=["model", "ema1", "ema2"],
        help="Which checkpoint weights to convert.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="heun",
        choices=["heun", "euler"],
        help="Default solver to store in the scheduler config.",
    )
    parser.add_argument("--t_eps", type=float, default=5e-2, help="Stability epsilon for velocity computation.")
    parser.add_argument("--safe_serialization", action="store_true", help="Save weights using safetensors.")
    parser.add_argument(
        "--id2label_path",
        type=str,
        default=None,
        help="Optional JSON file with id2label mapping to store in model_index.json.",
    )
    return parser.parse_args()


def _load_id2label(path: Optional[str]) -> Optional[Dict[int, str]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("id2label JSON must be an object mapping ids to labels.")
    return {int(k): str(v) for k, v in data.items()}


def main():
    args = get_args()

    transformer, metadata = PixelREPATransformer2DModel.from_pixelrepa_checkpoint(
        args.checkpoint_path,
        weights=args.weights,
    )
    scheduler = PixelREPAScheduler(t_eps=args.t_eps, solver=args.solver)
    id2label = _load_id2label(args.id2label_path)
    pipeline = PixelREPAPipeline(transformer=transformer, scheduler=scheduler, id2label=id2label)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.save_pretrained(str(output_dir), safe_serialization=args.safe_serialization)

    conversion_metadata = {
        "checkpoint_path": args.checkpoint_path,
        "weights": args.weights,
        "model_type": metadata.get("model_type"),
        "epoch": metadata.get("epoch"),
        "pixelrepa_args": metadata.get("source_args"),
    }
    (output_dir / "conversion_metadata.json").write_text(
        json.dumps(conversion_metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Diffusers PixelREPA model to: {output_dir}")


if __name__ == "__main__":
    main()
