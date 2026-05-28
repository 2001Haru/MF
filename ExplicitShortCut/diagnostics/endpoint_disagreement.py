import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class RecursiveImageDataset(Dataset):
    def __init__(
        self,
        root,
        image_size=256,
        max_images=None,
        label_map=None,
        list_max_images=False,
    ):
        self.root = Path(root)
        self.label_map = label_map
        suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        paths = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in suffixes:
                paths.append(p)
                if list_max_images and max_images is not None and len(paths) >= max_images:
                    break
        self.paths = sorted(paths)
        if max_images is not None and not list_max_images:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        rel_path = path.relative_to(self.root).as_posix()
        label = -1 if self.label_map is None else int(self.label_map[rel_path])
        return image, rel_path, label


def parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def load_sit_model(args, device):
    latent_size = args.resolution // 8
    block_kwargs = {"fused_attn": False, "qk_norm": False}

    if args.adapt_model:
        from arch.sit_adpt import SiT_models
    else:
        from arch.sit import SiT_models

    print(f"[info] Building model {args.model} (adapt_model={args.adapt_model})")
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        **block_kwargs,
    ).to(device)

    print(f"[info] Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint[args.ckpt_key]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_vae(device):
    local_path = "./ckpt/stabilityai/sd-vae-ft-ema"
    if os.path.exists(local_path):
        print(f"[info] Loading local VAE: {local_path}")
        vae = AutoencoderKL.from_pretrained(local_path).to(device)
    else:
        print("[info] Loading VAE from Hugging Face: stabilityai/sd-vae-ft-ema")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    return vae


def encode_latents(vae, images):
    if hasattr(vae, "_encode"):
        posterior = DiagonalGaussianDistribution(vae._encode(images))
        return posterior.sample()
    return vae.encode(images).latent_dist.sample()


def load_label_map(args):
    if args.label_mode != "json":
        return None

    label_json = Path(args.label_json) if args.label_json else Path(args.image_root) / "dataset.json"
    print(f"[info] Loading label json: {label_json}")
    with label_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    label_map = dict(payload["labels"])
    return {str(k).replace("\\", "/"): int(v) for k, v in label_map.items()}


@torch.no_grad()
def main(args):
    device = torch.device(args.device if args.device else "cuda")
    print(f"[info] Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    t_values = parse_float_list(args.t_values)
    r_fractions = parse_float_list(args.r_fractions)
    if any(t <= 0 or t > 1 for t in t_values):
        raise ValueError("--t-values must be in (0, 1].")
    if any(r < 0 or r > 1 for r in r_fractions):
        raise ValueError("--r-fractions must be in [0, 1].")

    label_map = load_label_map(args)
    print(f"[info] Scanning images under: {args.image_root}")
    dataset = RecursiveImageDataset(
        args.image_root,
        args.resolution,
        args.max_images,
        label_map=label_map,
        list_max_images=args.list_max_images,
    )
    print(f"[info] Found {len(dataset)} images for diagnosis")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = load_sit_model(args, device)
    vae = load_vae(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Output dir: {output_dir}")
    csv_path = output_dir / "endpoint_disagreement_samples.csv"
    json_path = output_dir / "endpoint_disagreement_summary.json"

    latents_scale = torch.tensor(
        [0.18125, 0.18125, 0.18125, 0.18125], device=device
    ).view(1, 4, 1, 1)
    latents_bias = torch.zeros_like(latents_scale)

    all_records = []
    per_t_disagreement = {str(t): [] for t in t_values}
    per_t_endpoint_error = {str(t): [] for t in t_values}
    per_t_true_endpoint_error_by_r = {
        str(t): {str(frac): [] for frac in r_fractions} for t in t_values
    }

    null_y_value = args.num_classes

    pbar = tqdm(dataloader, desc="Endpoint disagreement")
    for images, rel_paths, labels in pbar:
        images = images.to(device, non_blocking=True)
        x_data = encode_latents(vae, images)
        x_data = x_data * latents_scale + latents_bias

        noise = torch.randn_like(x_data)
        batch_size = x_data.shape[0]

        if args.label_mode == "null":
            y = torch.full((batch_size,), null_y_value, device=device, dtype=torch.long)
        elif args.label_mode == "random":
            y = torch.randint(0, args.num_classes, (batch_size,), device=device)
        elif args.label_mode == "json":
            y = labels.to(device, non_blocking=True).long()
        else:
            raise ValueError(f"Unsupported label mode: {args.label_mode}")

        for t_value in t_values:
            t = torch.full((batch_size,), t_value, device=device)
            x_t = (1.0 - t_value) * x_data + t_value * noise

            endpoint_preds = []
            per_r_endpoint_errors = []
            for frac in r_fractions:
                r_value = t_value * frac
                r = torch.full((batch_size,), r_value, device=device)
                u = model(x_t, r, t, y=y)
                # In this repo's linear path, data is at time 0. Therefore
                # extrapolating from t to the data endpoint uses x_t - t * u.
                x_hat_data = x_t - t.view(-1, 1, 1, 1) * u
                endpoint_preds.append(x_hat_data)

                err = ((x_hat_data - x_data) ** 2).mean(dim=(1, 2, 3))
                per_r_endpoint_errors.append(err)
                per_t_true_endpoint_error_by_r[str(t_value)][str(frac)].extend(
                    err.detach().cpu().tolist()
                )

            endpoint_preds = torch.stack(endpoint_preds, dim=0)
            endpoint_mean = endpoint_preds.mean(dim=0)
            disagreement = ((endpoint_preds - endpoint_mean) ** 2).mean(
                dim=(0, 2, 3, 4)
            )
            endpoint_error = ((endpoint_mean - x_data) ** 2).mean(dim=(1, 2, 3))

            disagreement_cpu = disagreement.detach().cpu().tolist()
            endpoint_error_cpu = endpoint_error.detach().cpu().tolist()
            per_r_error_cpu = [
                e.detach().cpu().tolist() for e in per_r_endpoint_errors
            ]

            per_t_disagreement[str(t_value)].extend(disagreement_cpu)
            per_t_endpoint_error[str(t_value)].extend(endpoint_error_cpu)

            for i, rel_path in enumerate(rel_paths):
                record = {
                    "path": rel_path,
                    "t": t_value,
                    "endpoint_disagreement": disagreement_cpu[i],
                    "mean_endpoint_mse_to_data": endpoint_error_cpu[i],
                }
                for j, frac in enumerate(r_fractions):
                    record[f"endpoint_mse_to_data_rfrac_{frac}"] = per_r_error_cpu[j][i]
                all_records.append(record)

    fieldnames = list(all_records[0].keys()) if all_records else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    summary = {
        "ckpt": args.ckpt,
        "ckpt_key": args.ckpt_key,
        "image_root": args.image_root,
        "num_images": len(dataset),
        "model": args.model,
        "adapt_model": args.adapt_model,
        "resolution": args.resolution,
        "label_mode": args.label_mode,
        "label_json": args.label_json,
        "num_classes": args.num_classes,
        "t_values": t_values,
        "r_fractions": r_fractions,
        "data_endpoint_time": 0.0,
        "latent_scale": 0.18125,
        "by_t": {},
        "outputs": {
            "sample_csv": str(csv_path),
            "summary_json": str(json_path),
        },
    }

    for t_value in t_values:
        t_key = str(t_value)
        summary["by_t"][t_key] = {
            "endpoint_disagreement": summarize(per_t_disagreement[t_key]),
            "mean_endpoint_mse_to_data": summarize(per_t_endpoint_error[t_key]),
            "endpoint_mse_to_data_by_r_fraction": {
                str(frac): summarize(per_t_true_endpoint_error_by_r[t_key][str(frac)])
                for frac in r_fractions
            },
        }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["by_t"], indent=2))
    print(f"Saved sample records to {csv_path}")
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose ESC endpoint disagreement across shortcut r values."
    )
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./diagnostics/endpoint_disagreement")
    parser.add_argument("--model", type=str, default="SiT-B/2")
    parser.add_argument("--adapt-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt-key", type=str, default="ema", choices=["ema", "model", "model_tgt"])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=1024)
    parser.add_argument("--list-max-images", action="store_true")
    parser.add_argument("--t-values", type=str, default="0.25,0.5,0.75,0.9")
    parser.add_argument("--r-fractions", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--label-mode", type=str, default="null", choices=["null", "random", "json"])
    parser.add_argument("--label-json", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(args)
