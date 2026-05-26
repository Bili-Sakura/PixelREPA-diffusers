import argparse
import sys
from pathlib import Path

import torch

try:
    from src.diffusers.pipelines.pixelrepa.pipeline_pixelrepa import PixelREPAPipeline
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.diffusers.pipelines.pixelrepa.pipeline_pixelrepa import PixelREPAPipeline


def get_args():
    parser = argparse.ArgumentParser(description="Sample a PixelREPA Diffusers model.")
    parser.add_argument("--model", type=str, required=True, help="Path to PixelREPA diffusers model.")
    parser.add_argument("--output", type=str, required=True, help="Output image path.")
    parser.add_argument("--class-label", type=str, required=True, help="Class label id or English label.")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of sampling steps.")
    parser.add_argument("--solver", type=str, default="heun", choices=["heun", "euler"], help="Solver to use.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Classifier-free guidance scale.")
    parser.add_argument("--guidance-interval-min", type=float, default=0.1, help="CFG interval minimum.")
    parser.add_argument("--guidance-interval-max", type=float, default=1.0, help="CFG interval maximum.")
    parser.add_argument("--noise-scale", type=float, default=None, help="Initial noise scale.")
    parser.add_argument("--t-eps", type=float, default=None, help="Numerical stability epsilon.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main():
    args = get_args()

    pipeline = PixelREPAPipeline.from_pretrained(args.model)
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)

    try:
        class_label = int(args.class_label)
    except ValueError:
        class_label = args.class_label

    result = pipeline(
        class_labels=class_label,
        guidance_scale=args.guidance_scale,
        guidance_interval_min=args.guidance_interval_min,
        guidance_interval_max=args.guidance_interval_max,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        sampling_method=args.solver,
        generator=generator,
        num_inference_steps=args.num_inference_steps,
    )

    image = result.images[0]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))
    print(f"Saved sample to: {output_path}")


if __name__ == "__main__":
    main()
