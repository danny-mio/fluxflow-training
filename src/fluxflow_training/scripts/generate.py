"""FluxFlow image generation script."""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import safetensors.torch
import torch
from diffusers import DPMSolverMultistepScheduler
from fluxflow.models import (
    BertTextEncoder,
    FluxCompressor,
    FluxExpander,
    FluxFlowProcessor,
    FluxPipeline,
)
from fluxflow.utils import generate_latent_images
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from transformers import AutoTokenizer

from fluxflow_training.data import TextImageDataset, collate_fn_generate
from fluxflow_training.training import get_device


def generate(args):
    """Generate images from text prompts using trained FluxFlow model."""
    device = get_device()
    print(f"Using device: {device}")

    # Load tokenizer (uses cache if present, otherwise downloads)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, cache_dir="./_cache", local_files_only=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Initialize models
    text_encoder = BertTextEncoder(embed_dim=args.text_embedding_dim)
    diffuser = FluxPipeline(
        FluxCompressor(d_model=args.vae_dim),
        FluxFlowProcessor(d_model=args.feature_maps_dim, vae_dim=args.vae_dim),
        FluxExpander(d_model=args.vae_dim),
    )

    # Load checkpoint
    print(f"Loading model from {args.model_checkpoint}")
    state_dict = safetensors.torch.load_file(args.model_checkpoint)
    diffuser.load_state_dict(
        {k.replace("diffuser.", ""): v for k, v in state_dict.items() if k.startswith("diffuser.")},
        strict=False,
    )
    text_encoder.load_state_dict(
        {
            k.replace("text_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("text_encoder.")
        },
        strict=False,
    )

    diffuser.to(device).eval()
    text_encoder.to(device).eval()

    # Load dataset (text prompts from .txt files)
    dataset = TextImageDataset(
        data_path=args.text_prompts_path,
        captions_file=None,
        tokenizer_name=args.tokenizer_name,
        transform=None,
        generate_mode=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn_generate,
    )

    os.makedirs(args.output_path, exist_ok=True)
    print(f"Generating images to {args.output_path}")

    # Generate images
    with torch.no_grad():
        for i, (file_names, input_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
            text_embeddings = text_encoder(input_ids, attention_mask=attention_mask)

            B = text_embeddings.size(0)
            size = args.img_size

            # Create random latent
            z_img = (torch.rand((B, 3, size, size), device=device) * 2) - 1
            latent_z = diffuser.compressor(z_img)

            img_seq = latent_z[:, :-1, :].contiguous()
            hw_vec = latent_z[:, -1:, :].contiguous()

            noise_img = torch.randn_like(img_seq)

            # Noise schedule
            scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)
            scheduler.set_timesteps(args.ddim_steps, device=device)  # type: ignore[arg-type]

            t = torch.randint(0, 1000, (B,), device=device)
            noised_img = scheduler.add_noise(img_seq, noise_img, t)  # type: ignore
            noised_latent = torch.cat([noised_img, hw_vec], dim=1)

            # Denoise (with optional CFG)
            if args.use_cfg and args.guidance_scale != 1.0:
                # Use CFG-guided generation
                from fluxflow_training.training.cfg_inference import generate_with_cfg

                decoded_images = generate_with_cfg(
                    diffuser=diffuser,
                    text_embeddings=text_embeddings,
                    guidance_scale=args.guidance_scale,
                    img_size=size,
                    num_steps=args.ddim_steps,
                    batch_size=B,
                    device=device,
                )
            else:
                # Standard generation (no CFG)
                denoised_latent = generate_latent_images(
                    batch_z=noised_latent,
                    text_embeddings=text_embeddings,
                    diffuser=diffuser,
                    steps=args.ddim_steps,
                    prediction_type="v_prediction",
                )
                # Decode and save
                decoded_images = diffuser.expander(denoised_latent)

            # Save generated images
            for idx, image in enumerate(decoded_images):
                file_name = os.path.splitext(file_names[idx])[0]
                save_path = os.path.join(args.output_path, f"{file_name}_gen.webp")
                # Expander outputs in [-1, 1] range, normalize to [0, 1] for saving
                save_image(image, save_path, normalize=True, value_range=(-1, 1))
                print(f"Saved: {save_path}")

    print(f"Generation complete! Images saved to {args.output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from text prompts using FluxFlow")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--text_prompts_path",
        type=str,
        required=True,
        help="Directory containing .txt files with prompts",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs",
        help="Directory to save generated images",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--workers", type=int, default=1, help="Number of data loading workers")
    parser.add_argument("--img_size", type=int, default=512, help="Generated image size")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--text_embedding_dim", type=int, default=1024, help="Text embedding dimension"
    )
    parser.add_argument("--vae_dim", type=int, default=128, help="VAE latent dimension")
    parser.add_argument(
        "--feature_maps_dim",
        type=int,
        default=128,
        help="Flow processor feature dimension",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (1.0=no guidance, 3-9=typical range)",
    )
    parser.add_argument(
        "--use_cfg",
        action="store_true",
        help="Enable classifier-free guidance (requires model trained with CFG)",
    )
    return parser.parse_args()


def main():
    """Main entry point for the generation script."""
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
