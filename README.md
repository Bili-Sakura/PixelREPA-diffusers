# PixelREPA Diffusers Refactor

This repository is now fully organized around a Diffusers-style package layout, following the same migration pattern used in [Bili-Sakura/JiT-diffusers](https://github.com/Bili-Sakura/JiT-diffusers).

Legacy standalone training/evaluation codepaths have been removed so the tree is focused on reusable Diffusers components and checkpoint conversion.

## Package layout

- `src/diffusers/models/transformers/transformer_pixelrepa.py`: `PixelREPATransformer2DModel` (`ModelMixin`/`ConfigMixin`) class-conditional transformer.
- `src/diffusers/schedulers/scheduling_pixelrepa.py`: `PixelREPAScheduler` with Euler/Heun flow-matching updates.
- `src/diffusers/pipelines/pixelrepa/pipeline_pixelrepa.py`: `PixelREPAPipeline` with classifier-free guidance and native-resolution sampling.
- `scripts/convert_pixelrepa_to_diffusers.py`: converts legacy PixelREPA training checkpoints to Diffusers model directories.
- `scripts/convert_diffusers_to_pixelrepa.py`: converts Diffusers PixelREPA models back to legacy checkpoint format (JiT backbone only).
- `scripts/sample_pixelrepa.py`: single-image sampling script for converted models.

## Convert a checkpoint

```bash
python scripts/convert_pixelrepa_to_diffusers.py \
  --checkpoint_path checkpoints/checkpoint-last.pth \
  --output_dir pixelrepa-diffusers \
  --weights ema1 \
  --safe_serialization
```

The generated `conversion_metadata.json` includes Diffusers-style fields and PixelREPA legacy aliases for compatibility.

## Convert back to legacy checkpoint

```bash
python scripts/convert_diffusers_to_pixelrepa.py \
  --model_path pixelrepa-diffusers \
  --output_path checkpoint-converted.pth \
  --ema_mode copy_to_both
```

This conversion recreates the JiT backbone weights only; Masked Transformer Adapter and REPA encoder weights are not included.

## Sample

```bash
python scripts/sample_pixelrepa.py \
  --model pixelrepa-diffusers \
  --output demo.png \
  --class-label 207 \
  --num-inference-steps 50 \
  --solver heun
```

## Notes

- This repository is intended for Diffusers integration and checkpoint conversion workflows.
- For direct upstreaming, copy files under `src/diffusers` into matching paths in `huggingface/diffusers` and register lazy imports there.

## Citation

```bibtex
@misc{shin2026pixelrepa,
      title={Representation Alignment for Just Image Transformers is not Easier than You Think}, 
      author={Jaeyo Shin and Jiwook Kim and Hyunjung Shim},
      year={2026},
      eprint={2603.14366},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.14366}, 
}
```
