# type: ignore
import argparse
import datetime
import json
import numpy as np
import os
import time
import pickle
import io
from tqdm import tqdm
import torch.distributed as dist

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import lmdb
from torchvision.datasets import ImageFolder

from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size

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

class ImageFolderDataset(ImageFolder):
    """
    ImageFolder dataset with additional functionality similar to LMDBImageNetReader.
    
    Args:
        folder_dir (str): Path to the image folder directory.
        transform (callable, optional): Transform to apply to images.
        flip_prob (float): Probability of horizontal flip.
        target_transform (callable, optional): Transform to apply to labels.
        loader (callable, optional): Function to load images.
        is_valid_file (callable, optional): Function to check if file is valid.
    """
    def __init__(self, folder_dir, transform=None, flip_prob=0.5, **kwargs):
        super().__init__(folder_dir, transform=transform, **kwargs)
        self.flip_prob = flip_prob
        
    def __getitem__(self, index):
        """
        Get item with optional horizontal flip.
        
        Returns:
            tuple: (image, label, filename, index)
        """
        path, label = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        # Get filename from path
        filename = os.path.relpath(path, self.root)
        
        return sample, label, filename, index


class JsonLabelImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder_dir, label_json, transform=None):
        self.root = folder_dir
        self.transform = transform

        with open(label_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        records = metadata["labels"]

        self.samples = []
        for rel_path, label in records:
            abs_path = os.path.join(self.root, rel_path)
            if os.path.isfile(abs_path):
                self.samples.append((rel_path, int(label)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images from {label_json} were found under {folder_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        path = os.path.join(self.root, filename)
        sample = Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, filename, index


def encode_moments(vae, images):
    if hasattr(vae, "_encode"):
        return DiagonalGaussianDistribution(vae._encode(images)).parameters
    return vae.encode(images).latent_dist.parameters


def process_batch(args, vae, device, images, labels, filenames, original_indices, env):
    images = images.to(device)
    
    with torch.no_grad():
        moments = encode_moments(vae, images)
        moments_flip = encode_moments(vae, images.flip(dims=[3]))
    
    try:
        with env.begin(write=True) as txn:
            for i, (label, filename, orig_idx) in enumerate(zip(labels, filenames, original_indices)):
                data = {
                    'moments': moments[i].cpu().numpy(),
                    'moments_flip': moments_flip[i].cpu().numpy(),
                    'label': label.item(),
                    'filename': filename
                }
                
                txn.put(f'{orig_idx}'.encode(), pickle.dumps(data))
    except lmdb.Error as e:
        print(f"LMDB error during batch processing: {e}")
        return 0
        
    return len(images)

def preprocess_latents(args):

    local_rank, rank, world_size = init_distributed()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    cudnn.benchmark = True
    if args.local_vae_path and os.path.exists(args.local_vae_path):
        if rank == 0:
            print(f"Loading local VAE from {args.local_vae_path}")
        vae = AutoencoderKL.from_pretrained(args.local_vae_path).to(device)
    else:
        if rank == 0:
            print(f"Loading VAE from Hugging Face: {args.vae_name}")
        vae = AutoencoderKL.from_pretrained(args.vae_name).to(device)
    vae.eval().requires_grad_(False)
    
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    if rank == 0:
        print(f"Loading source dataset from {args.folder_dir}")
    if args.label_json:
        if rank == 0:
            print(f"Using labels from {args.label_json}")
        dataset = JsonLabelImageDataset(args.folder_dir, args.label_json, transform=transform)
    else:
        dataset = ImageFolderDataset(args.folder_dir, transform=transform)
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    os.makedirs(os.path.dirname(args.target_lmdb) if os.path.dirname(args.target_lmdb) else '.', exist_ok=True)
    
    map_size = 1024 * 1024 * 1024 * args.lmdb_size_gb
    if rank == 0:
        print(f"Creating target LMDB at {args.target_lmdb} with size {args.lmdb_size_gb}GB")
    
    env = None
    try:
        env = lmdb.open(args.target_lmdb, map_size=map_size, max_readers=world_size*2, max_spare_txns=world_size*2)
        
        total_processed = 0
        start_time = time.time()
        
        pbar = None
        if rank == 0:
            pbar = tqdm(total=len(dataloader), desc=f"Processing")
        
        for batch_idx, (images, labels, filenames, original_indices) in enumerate(dataloader):
            num_processed = process_batch(
                args, vae, device, images, labels, filenames, original_indices, env
            )
            
            total_processed += num_processed
            if pbar is not None:
                pbar.update(1)
        
        if pbar is not None:
            pbar.close()
        
        
        if rank == 0:
            if dist.is_initialized():
                dist.barrier()
            try:
                with env.begin(write=True) as txn:
                    txn.put('num_samples'.encode(), str(len(dataset)).encode())
                    txn.put('created_at'.encode(), str(datetime.datetime.now()).encode())
            except lmdb.Error as e:
                print(f"Error writing metadata: {e}")
        elif dist.is_initialized():
            dist.barrier()
        
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if rank == 0:
            print(f'Preprocessing completed in {total_time_str}')
            print(f'Each process processed approximately {total_processed} samples')
            print(f'Target LMDB saved at: {args.target_lmdb}')
    
    except Exception as e:
        print(f"Process {rank} encountered error: {e}")
        if env is not None:
            try:
                env.close()
            except:
                pass
        raise
    
    finally:
        if env is not None:
            try:
                env.sync()
                env.close()
                if rank == 0:
                    print("LMDB environment closed successfully")
            except Exception as e:
                print(f"Process {rank} error closing LMDB: {e}")

    if dist.is_initialized():
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess ImageNet to VAE latents')
    parser.add_argument('--folder_dir', type=str, default='./image_data/ImageNet/train',
                        help='Path to source ImageNet folder')
    parser.add_argument('--target_lmdb', type=str, default='./image_data/imagenet_vq_lmdb/train',
                        help='Path to save target latents LMDB')
    parser.add_argument('--label_json', type=str, default='',
                        help='Optional dataset.json with labels in [[relative_path, label], ...] format')
    parser.add_argument('--vae_name', type=str, default='stabilityai/sd-vae-ft-ema',
                        help='Hugging Face VAE name. Use stabilityai/sd-vae-ft-ema for ESC.')
    parser.add_argument('--local_vae_path', type=str, default='./ckpt/stabilityai/sd-vae-ft-ema',
                        help='Local VAE directory. Used before downloading from Hugging Face.')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for preprocessing')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--lmdb_size_gb', type=int, default=300,
                        help='LMDB size in GB')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # if 'RANK' not in os.environ:
    #     raise RuntimeError("Please use torchrun to run this script. Example: torchrun --nproc_per_node=8 main_cache.py ...")
    
    preprocess_latents(args)
