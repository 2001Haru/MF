import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
    def __init__(self, root, image_size=256, max_images=None, label_map=None, shuffle=False, seed=0):
        self.root = Path(root)
        self.label_map = label_map
        suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        if label_map is not None:
            paths = [
                self.root / rel_path for rel_path in label_map.keys()
                if Path(rel_path).suffix.lower() in suffixes
            ]
        else:
            paths = [
                p for p in self.root.rglob("*")
                if p.is_file() and p.suffix.lower() in suffixes
            ]

        self.paths = sorted(paths)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.paths)
        if max_images is not None:
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


def parse_triples(value):
    triples = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [float(x.strip()) for x in item.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid triple '{item}'. Expected t:s:r.")
        t, s, r = parts
        if not (0 <= r < s < t <= 1):
            raise ValueError(f"Expected 0 <= r < s < t <= 1, got {item}.")
        triples.append((t, s, r))
    if not triples:
        raise ValueError("No valid triples provided.")
    return triples


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


def load_label_map(args):
    if args.label_mode != "json":
        return None
    label_json = Path(args.label_json) if args.label_json else Path(args.image_root) / "dataset.json"
    print(f"[info] Loading label json: {label_json}")
    with label_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    label_map = dict(payload["labels"])
    return {str(k).replace("\\", "/"): int(v) for k, v in label_map.items()}


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
    model.load_state_dict(checkpoint[args.ckpt_key])
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


def make_time(batch_size, value, device):
    return torch.full((batch_size,), float(value), device=device)


def mse_per_sample(x, y):
    return ((x - y) ** 2).mean(dim=(1, 2, 3))


@torch.no_grad()
def main(args):
    device = torch.device(args.device if args.device else "cuda")
    print(f"[info] Using device: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    triples = parse_triples(args.triples)
    label_map = load_label_map(args)

    print(f"[info] Preparing image list under: {args.image_root}")
    dataset = RecursiveImageDataset(
        args.image_root,
        image_size=args.resolution,
        max_images=args.max_images,
        label_map=label_map,
        shuffle=args.shuffle,
        seed=args.seed,
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

    latents_scale = torch.tensor(
        [0.18125, 0.18125, 0.18125, 0.18125], device=device
    ).view(1, 4, 1, 1)
    latents_bias = torch.zeros_like(latents_scale)

    null_y_value = args.num_classes
    records = []
    semigroup_stats = {f"{t}:{s}:{r}": {"gap": [], "gap_rel": []} for t, s, r in triples}
    fit_stats = {}

    def add_fit(key, values):
        fit_stats.setdefault(key, []).extend(values.detach().cpu().tolist())

    pbar = tqdm(dataloader, desc="Shortcut consistency")
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

        for t_value, s_value, r_value in triples:
            t = make_time(batch_size, t_value, device)
            s = make_time(batch_size, s_value, device)
            r = make_time(batch_size, r_value, device)

            x_t = (1.0 - t_value) * x_data + t_value * noise
            x_s_true = (1.0 - s_value) * x_data + s_value * noise
            x_r_true = (1.0 - r_value) * x_data + r_value * noise

            u_tr = model(x_t, r, t, y=y)
            x_r_one = x_t + (r_value - t_value) * u_tr

            u_ts = model(x_t, s, t, y=y)
            x_s_pred = x_t + (s_value - t_value) * u_ts

            u_sr_pred_path = model(x_s_pred, r, s, y=y)
            x_r_two = x_s_pred + (r_value - s_value) * u_sr_pred_path

            gap = mse_per_sample(x_r_one, x_r_two)
            denom = mse_per_sample(x_r_one, x_t).clamp_min(args.eps)
            gap_rel = gap / denom

            u_cond_tr = (x_t - x_r_true) / (t_value - r_value)
            u_cond_ts = (x_t - x_s_true) / (t_value - s_value)
            u_cond_sr = (x_s_true - x_r_true) / (s_value - r_value)

            fit_tr = mse_per_sample(u_tr, u_cond_tr)
            fit_ts = mse_per_sample(u_ts, u_cond_ts)
            # Target-fit is measured on the true path state x_s, not the
            # model-predicted x_s used in the semigroup two-step path.
            u_sr_true_path = model(x_s_true, r, s, y=y)
            fit_sr = mse_per_sample(u_sr_true_path, u_cond_sr)

            triple_key = f"{t_value}:{s_value}:{r_value}"
            semigroup_stats[triple_key]["gap"].extend(gap.detach().cpu().tolist())
            semigroup_stats[triple_key]["gap_rel"].extend(gap_rel.detach().cpu().tolist())
            add_fit(f"{t_value}->{r_value}", fit_tr)
            add_fit(f"{t_value}->{s_value}", fit_ts)
            add_fit(f"{s_value}->{r_value}", fit_sr)

            gap_cpu = gap.detach().cpu().tolist()
            gap_rel_cpu = gap_rel.detach().cpu().tolist()
            fit_tr_cpu = fit_tr.detach().cpu().tolist()
            fit_ts_cpu = fit_ts.detach().cpu().tolist()
            fit_sr_cpu = fit_sr.detach().cpu().tolist()

            for i, rel_path in enumerate(rel_paths):
                records.append({
                    "path": rel_path,
                    "label": int(labels[i].item()) if torch.is_tensor(labels) else int(labels[i]),
                    "t": t_value,
                    "s": s_value,
                    "r": r_value,
                    "semigroup_gap": gap_cpu[i],
                    "semigroup_gap_rel": gap_rel_cpu[i],
                    "target_fit_mse_t_to_r": fit_tr_cpu[i],
                    "target_fit_mse_t_to_s": fit_ts_cpu[i],
                    "target_fit_mse_s_to_r": fit_sr_cpu[i],
                })

    sample_csv = output_dir / "shortcut_consistency_samples.csv"
    summary_json = output_dir / "shortcut_consistency_summary.json"

    with sample_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(records[0].keys()) if records else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

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
        "triples": [{"t": t, "s": s, "r": r} for t, s, r in triples],
        "latent_scale": 0.18125,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "semigroup": {},
        "target_fit": {},
        "outputs": {
            "sample_csv": str(sample_csv),
            "summary_json": str(summary_json),
        },
    }

    for key, values in semigroup_stats.items():
        summary["semigroup"][key] = {
            "gap": summarize(values["gap"]),
            "gap_rel": summarize(values["gap_rel"]),
        }
    for key, values in sorted(fit_stats.items()):
        summary["target_fit"][key] = summarize(values)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "semigroup": summary["semigroup"],
        "target_fit": summary["target_fit"],
    }, indent=2))
    print(f"Saved sample records to {sample_csv}")
    print(f"Saved summary to {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose ESC shortcut semigroup gap and conditional target-fit error."
    )
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--label-mode", type=str, default="json", choices=["json", "null", "random"])
    parser.add_argument("--label-json", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./diagnostics/shortcut_consistency")
    parser.add_argument("--model", type=str, default="SiT-B/2")
    parser.add_argument("--adapt-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt-key", type=str, default="ema", choices=["ema", "model", "model_tgt"])
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=1024)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--triples",
        type=str,
        default="0.5:0.25:0,0.75:0.5:0.25,0.9:0.75:0.5,0.9:0.5:0",
        help="Comma-separated t:s:r triples with 0 <= r < s < t <= 1.",
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default=None)
    main(parser.parse_args())
