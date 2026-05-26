from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from ...models.transformers.transformer_pixelrepa import PixelREPATransformer2DModel
from ...schedulers.scheduling_pixelrepa import PixelREPAScheduler

RECOMMENDED_NOISE_BY_SIZE = {
    256: 1.0,
    512: 2.0,
}


class PixelREPAPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"

    def __init__(
        self,
        transformer: PixelREPATransformer2DModel,
        scheduler: Optional[PixelREPAScheduler] = None,
        id2label: Optional[Dict[Union[int, str], str]] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler or PixelREPAScheduler())
        self._id2label = self._normalize_id2label(id2label)
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        scheduler_kwargs = model_kwargs.pop("scheduler_kwargs", {})
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"

        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            if transformer_subfolder is not None:
                transformer_path = str(base_path / transformer_subfolder)
            else:
                transformer_path = pretrained_model_name_or_path

            transformer = PixelREPATransformer2DModel.from_pretrained(transformer_path, **model_kwargs)
            try:
                scheduler = PixelREPAScheduler.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=scheduler_subfolder,
                    **scheduler_kwargs,
                )
            except Exception:
                scheduler = PixelREPAScheduler(**scheduler_kwargs)

            id2label = cls._read_id2label_from_model_index(str(base_path))
            return cls(
                transformer=transformer,
                scheduler=scheduler,
                id2label=id2label,
            )

    def _ensure_labels_loaded(self) -> None:
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    @staticmethod
    def _normalize_id2label(id2label: Optional[Dict[Union[int, str], str]]) -> Dict[int, str]:
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _read_id2label_from_model_index(variant_path: Optional[str]) -> Dict[int, str]:
        if not variant_path:
            return {}
        variant_dir = Path(variant_path).resolve()
        model_index_path = variant_dir / "model_index.json"
        if not model_index_path.exists():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict):
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _build_label2id(id2label: Dict[int, str]) -> Dict[str, int]:
        label2id: Dict[str, int] = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    @property
    def id2label(self) -> Dict[int, str]:
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: Union[str, List[str]]) -> List[int]:
        self._ensure_labels_loaded()
        label2id = self.labels
        if not label2id:
            raise ValueError(
                "No English labels loaded. Ensure `id2label` exists in model_index.json."
            )

        if isinstance(label, str):
            label = [label]

        missing = [item for item in label if item not in label2id]
        if missing:
            preview = ", ".join(list(label2id.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [label2id[item] for item in label]

    def _normalize_class_labels(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
    ) -> List[int]:
        if isinstance(class_labels, int):
            return [class_labels]

        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)

        if class_labels and isinstance(class_labels[0], str):
            return self.get_label_ids(class_labels)

        return list(class_labels)

    def _predict_velocity(
        self,
        z_value: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
    ) -> torch.Tensor:
        t = torch.as_tensor(t, device=z_value.device, dtype=z_value.dtype)
        if do_classifier_free_guidance:
            z_in = torch.cat([z_value, z_value], dim=0)
            labels = torch.cat([class_labels, class_null], dim=0)
        else:
            z_in = z_value
            labels = class_labels

        t_batch = t.flatten().expand(z_in.shape[0])
        x_pred = self.transformer(z_in, timestep=t_batch, class_labels=labels).sample
        v = self.scheduler.velocity_from_prediction(z_in, x_pred, t)

        if not do_classifier_free_guidance:
            return v

        v_cond, v_uncond = v.chunk(2, dim=0)
        interval_mask = (t < guidance_interval_max) & (t > guidance_interval_min)
        scale = torch.where(
            interval_mask,
            torch.tensor(guidance_scale, device=z_value.device, dtype=z_value.dtype),
            torch.tensor(1.0, device=z_value.device, dtype=z_value.dtype),
        )
        return v_uncond + scale * (v_cond - v_uncond)

    def _run_sampler(
        self,
        latents: torch.Tensor,
        class_labels: torch.Tensor,
        class_null: torch.Tensor,
        num_inference_steps: int,
        do_classifier_free_guidance: bool,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
        sampling_method: str,
    ) -> torch.Tensor:
        device = latents.device
        self.scheduler.set_timesteps(num_inference_steps, device=device, solver=sampling_method)
        timesteps = self.scheduler.timesteps

        for i in self.progress_bar(range(num_inference_steps - 1)):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            v = self._predict_velocity(
                latents,
                t,
                class_labels,
                class_null,
                do_classifier_free_guidance,
                guidance_scale,
                guidance_interval_min,
                guidance_interval_max,
            )

            if sampling_method == "heun":
                latents_euler = latents + (t_next - t) * v
                v_next = self._predict_velocity(
                    latents_euler,
                    t_next,
                    class_labels,
                    class_null,
                    do_classifier_free_guidance,
                    guidance_scale,
                    guidance_interval_min,
                    guidance_interval_max,
                )
                latents = self.scheduler.step(v, t, latents, model_output_next=v_next).prev_sample
            else:
                latents = self.scheduler.step(v, t, latents).prev_sample

        t = timesteps[-2]
        t_next = timesteps[-1]
        v = self._predict_velocity(
            latents,
            t,
            class_labels,
            class_null,
            do_classifier_free_guidance,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
        )
        return latents + (t_next - t) * v

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, str, List[Union[int, str]]],
        guidance_scale: Optional[float] = None,
        guidance_interval_min: float = 0.1,
        guidance_interval_max: float = 1.0,
        noise_scale: Optional[float] = None,
        t_eps: Optional[float] = None,
        sampling_method: Optional[str] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        solver = sampling_method or self.scheduler.config.solver
        if solver not in {"heun", "euler"}:
            raise ValueError("sampling_method must be one of: 'heun', 'euler'.")
        if output_type not in {"pil", "np", "pt"}:
            raise ValueError("output_type must be one of: 'pil', 'np', 'pt'.")

        if t_eps is not None:
            self.scheduler.register_to_config(t_eps=t_eps)

        class_label_ids = self._normalize_class_labels(class_labels)
        do_classifier_free_guidance = guidance_scale is not None and guidance_scale > 1.0

        batch_size = len(class_label_ids)
        image_size = int(self.transformer.config.sample_size)
        channels = int(self.transformer.config.in_channels)
        null_class_val = int(
            getattr(self.transformer.config, "num_classes", getattr(self.transformer.config, "num_class_embeds", 1000))
        )

        if guidance_scale is None:
            guidance_scale = 1.0
        if noise_scale is None:
            noise_scale = RECOMMENDED_NOISE_BY_SIZE.get(image_size, 1.0)

        latents = randn_tensor(
            shape=(batch_size, channels, image_size, image_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        ) * noise_scale

        class_labels_t = torch.tensor(class_label_ids, device=self._execution_device, dtype=torch.long).reshape(-1)
        class_labels_t = class_labels_t.clamp(0, null_class_val - 1)
        class_null = torch.full_like(class_labels_t, null_class_val)

        latents = self._run_sampler(
            latents,
            class_labels_t,
            class_null,
            num_inference_steps,
            do_classifier_free_guidance,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
            solver,
        )

        images_pt = ((latents.float().clamp(-1, 1) + 1.0) / 2.0).cpu()
        if output_type == "pt":
            images = images_pt
        elif output_type == "np":
            images = images_pt.permute(0, 2, 3, 1).numpy()
        else:
            images = self.numpy_to_pil(images_pt.permute(0, 2, 3, 1).numpy())

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


PixelREPAPipelineOutput = ImagePipelineOutput
