## Quick Start

```bash
conda env create -f environment.yml
conda activate My_DiT
```

The environment ships with `jax[cpu]`, `flax`, and `optax` so you can develop on CPU and later move the same code to GPU/TPUs.

## Project Overview

This repository hosts a JAX/Flax implementation of the Diffusion Transformer (DiT) architecture, targeting the DiT-S variant (~33M parameters) described in *Scalable Diffusion Models with Transformers*. The end goal is to train on ImageNet-256 and reach FID-10K < 20 while staying within the DiT-S parameter budget.

## Repository Map

- `models/patch_embed.py` – Convolutional patch embedding with learned positional tokens.
- `models/attention.py` – Multi-head self-attention with shared QKV projection and dropout controls.
- `models/transformer.py` – DiT blocks with adaptive LayerNorm modulation and gated residual paths.
- `models/dit.py` – End-to-end DiT backbone, including timestep and optional class-conditioning.
- `utils/time_embedding.py` – Sinusoidal timestep embeddings followed by a small MLP.
- `test_shape.py` – Minimal smoke test that instantiates the model and checks tensor shapes.
- `DiT.ipynb` – Scratch notebook for experimentation (pair with the code for iteration).

## How the Model Works

1. **Patch Embedding**: Images are split by a strided convolution into a sequence of patch tokens (`PatchEmbed`). Learnable positional embeddings retain spatial structure.
2. **Conditioning Signals**: Timestep embeddings (`TimeEmbedding`) and optional class embeddings (`LabelEmbedding`) are summed to produce a conditioning vector shared across every transformer block. This aligns with classifier-free guidance when dropout is enabled on class labels.
3. **DiT Blocks**: Each block applies adaptive LayerNorm to inject the conditioning signal, runs multi-head attention and an MLP, then gates each residual branch via learnable `tanh` gates. The implementation mirrors DiT-S, keeping the parameter count in budget.
4. **Final Layer**: An adaptive LayerNorm (zero-initialised) plus linear projection reconstructs patch pixels, which are reshaped back into the image grid (`FinalLayer`).

`test_shape.py` shows the minimal usage pattern:

```python
model = DiT(image_size=64, patch_size=8, dimension=256, depth=4, num_heads=8)
params = model.init(key, dummy_x, dummy_t)
pred = model.apply(params, dummy_x, dummy_t)
```

When adding classifier conditioning pass `num_classes` to `DiT` and provide `class_labels` at call time.

## Training Checklist

- **Dataset**: Start with ImageNet-100 or ImageNet-256x256 depending on compute. Preprocess images into TFRecords or NumPy arrays for fast streaming.
- **Objective**: Use standard diffusion training (predict noise or velocity) with variance-preserving schedules. Ensure the timestep sampler matches your diffusion process.
- **Optimizer**: AdamW or RMSProp with warmup + cosine decay works well. Track EMA weights; FID typically uses EMA parameters.
- **Regularisation**: The model exposes dropout knobs (`dropout`, `attn_dropout`, `class_dropout_prob`) for stability and classifier-free guidance.
- **Evaluation**: Compute FID-10K using generated samples vs. ImageNet validation features (Inception-V3). Automate periodic evaluation to monitor convergence.

## Running Training

The `train` package wraps the DiT model with a diffusion loss, optimiser, EMA tracking, and checkpointing. You supply an iterator that yields batches shaped like:

```python
{"images": np.ndarray[B, H, W, 3], "labels": np.ndarray[B]}  # labels optional
```

Example skeleton:

```python
from train import DiffusionConfig, TrainingConfig, train
from models.dit import DiT

model_cfg = dict(
    image_size=256,
    patch_size=16,
    dimension=1152,
    depth=28,
    num_heads=16,
    mlp_ratio=4,
    channels=3,
    num_classes=1000,
    class_dropout_prob=0.1,
)

training_cfg = TrainingConfig(
    total_steps=250_000,
    learning_rate=1e-4,
    grad_clip_norm=1.0,
    checkpoint_every=5_000,
    workdir="./checkpoints/dit_s",
)

diffusion_cfg = DiffusionConfig(timesteps=1000, beta_schedule="cosine", prediction_type="eps")

state = train(model_cfg, training_cfg, diffusion_cfg, train_iterator=my_dataloader())
```

Hook `my_dataloader()` to your input pipeline (e.g., TFDS `imagenet2012`, WebDataset shards, or custom NumPy loaders). Use the returned `state.ema_params` for sampling checkpoints.

## Tips for Hitting FID < 20

1. Increase batch size or accumulate gradients to stabilise updates; DiT-S benefits from high effective batch.
2. Tune learning rate and EMA decay jointly—too aggressive learning rates quickly degrade FID.
3. Experiment with noise prediction vs. `v`-prediction. The latter can improve sample quality in practice.
4. Keep augmentation light; focus on precision in the input pipeline to avoid losing colour statistics.

## References

- Peebles & Xie, *Scalable Diffusion Models with Transformers*, 2022.
- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020.
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, 2021.

## Communication & Reporting

- Invite GitHub user `@r01566525` to your working repository.
- Send weekly reports (≤3 pages) to `r01566525@gmail.com`, noting any AI assistance or external code references.
- Reach out via the same email to schedule discussions or request cluster support.
