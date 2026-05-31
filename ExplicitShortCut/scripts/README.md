## Training ESC from Scratch
Training ESC from scratch with SiT-B/2 with class-consistent mini-batching, run the following
```bash
accelerate launch --multi_gpu \
    train.py \
    --exp-name "esc-b2-cc" \
    --output-dir "exp" \
    --data-dir "YOUR/DESTINATION/LMDB/PATH" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 512 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240 \
    --path-type "linear" \
    --loss-type "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 1.0 \
    --cfg-kappa 0.5 \
    --cfg-min-t 0.0 \
    --cfg-max-t 1.0 \
    --variational-adaptive-weight \
    --grad-warmup-steps 0 \
    --use-vplug \
    --vplug-prob 0.5 \
    --term-zero-steps 20000 \
    --class-consist \
    --no-debug
```

Or without class-consistent mini-batching:
```bash
accelerate launch --multi_gpu \
    train.py \
    --exp-name "esc-b2-nocc" \
    --output-dir "exp" \
    --data-dir "YOUR/DESTINATION/LMDB/PATH" \
    --model "SiT-B/2" \
    --resolution 256 \
    --batch-size 512 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240 \
    --path-type "linear" \
    --loss-type "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 1.0 \
    --cfg-kappa 0.5 \
    --cfg-min-t 0.0 \
    --cfg-max-t 1.0 \
    --variational-adaptive-weight \
    --grad-warmup-steps 0 \
    --use-vplug \
    --vplug-prob 0.5 \
    --term-zero-steps 20000 \
    --no-class-consist \
    --no-debug
```

Training ESC from scratch with SiT-XL/2 with class-consistent mini-batching, run the following
```bash
accelerate launch --multi_gpu \
    train.py \
    --exp-name "esc-xl-cc" \
    --output-dir "exp" \
    --data-dir "YOUR/DESTINATION/LMDB/PATH" \
    --model "SiT-XL/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240 \
    --path-type "linear" \
    --loss-type "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 0.2 \
    --cfg-kappa 0.92 \
    --cfg-min-t 0.0 \
    --cfg-max-t 0.75 \
    --variational-adaptive-weight \
    --grad-warmup-steps 0 \
    --use-vplug \
    --vplug-prob 0.2 \
    --term-zero-steps 20000 \
    --class-consist \
    --no-debug
```

Or without class-consistent mini-batching:
```bash
accelerate launch --multi_gpu \
    train.py \
    --exp-name "esc-xl-nocc" \
    --output-dir "exp" \
    --data-dir "YOUR/DESTINATION/LMDB/PATH" \
    --model "SiT-XL/2" \
    --resolution 256 \
    --batch-size 256 \
    --allow-tf32 \
    --mixed-precision "bf16" \
    --epochs 240 \
    --path-type "linear" \
    --loss-type "adaptive" \
    --time-sampler "logit_normal" \
    --time-mu -0.4 \
    --time-sigma 1.0 \
    --ratio-r-not-equal-t 0.25 \
    --adaptive-p 1.0 \
    --cfg-omega 0.2 \
    --cfg-kappa 0.92 \
    --cfg-min-t 0.0 \
    --cfg-max-t 0.75 \
    --variational-adaptive-weight \
    --grad-warmup-steps 0 \
    --use-vplug \
    --vplug-prob 0.2 \
    --term-zero-steps 20000 \
    --no-class-consist \
    --no-debug
```

## Euler MeanFlow Phase 1 Ablation

The JVP-free Euler MeanFlow experiments are additive and do not change the
official ESC training path. Run all four groups with the same data, seed, batch
size, and step budget:

```bash
DATA_DIR=/linxi/dataset/imagenet256_sd_ema_lmdb/train \
MAX_TRAIN_STEPS=20000 \
bash scripts/run_euler_phase1.sh bridge

DATA_DIR=/linxi/dataset/imagenet256_sd_ema_lmdb/train \
MAX_TRAIN_STEPS=20000 \
bash scripts/run_euler_phase1.sh euler-u

DATA_DIR=/linxi/dataset/imagenet256_sd_ema_lmdb/train \
MAX_TRAIN_STEPS=20000 \
bash scripts/run_euler_phase1.sh euler-u-weighted

DATA_DIR=/linxi/dataset/imagenet256_sd_ema_lmdb/train \
MAX_TRAIN_STEPS=20000 \
bash scripts/run_euler_phase1.sh euler-endpoint
```

The groups compare the original JVP MeanFlow bridge, EulerMF velocity
prediction, endpoint-equivalent `t^2` velocity weighting, and direct
endpoint-like prediction. The default script uses class labels without CFG
dropout or plug-in targets. Set `MULTI_GPU=0` for a single-GPU smoke test.
