# WGAN-GP: Improved Training of Wasserstein GANs

> **Paper**: [Gulrajani et al., 2017](https://arxiv.org/abs/1704.00028)  
> **TL;DR**: Replace weight clipping in WGAN with a gradient penalty — get stable training across almost any architecture, with no hyperparameter tuning.

---

## Table of Contents

- [Background](#background)
- [The Problem with Weight Clipping](#the-problem-with-weight-clipping)
- [The Fix: Gradient Penalty](#the-fix-gradient-penalty)
- [Algorithm](#algorithm)
- [Key Implementation Details](#key-implementation-details)
- [Results](#results)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Citation](#citation)

---

## Background

### Generative Adversarial Networks (GANs)

A GAN frames generation as a minimax game between two networks:

```
min  max  E_{x~P_r}[log D(x)]  +  E_{x̃~P_g}[log(1 - D(x̃))]
 G    D
```

- **Generator G** maps noise z → fake data x̃
- **Discriminator D** tries to distinguish real from fake

The core problem: when D is trained too well, it saturates and gives G vanishing gradients. Training becomes unstable.

---

### Wasserstein GAN (WGAN)

WGAN replaces the Jensen-Shannon divergence with the **Earth-Mover (Wasserstein-1) distance**:

```
W(P_r, P_g) = inf      E_{(x,y)~π} [ ‖x - y‖ ]
              π∈Π(P_r,P_g)
```

Think of it as the minimum "work" required to transform one distribution into the other — moving probability mass like dirt.

The WGAN objective (via Kantorovich-Rubinstein duality):

```
min  max       E_{x~P_r}[D(x)]  -  E_{x̃~P_g}[D(x̃)]
 G   D∈Lip-1
```

Where **D must be a 1-Lipschitz function**: `|D(x) - D(y)| ≤ ‖x - y‖` for all x, y.

**Why this is better:**
- W(P_r, P_g) is continuous and differentiable almost everywhere
- Critic loss directly correlates with sample quality (unlike GAN)
- No vanishing gradients when the critic is well-trained

---

## The Problem with Weight Clipping

The original WGAN enforces the Lipschitz constraint by clipping all critic weights to `[-c, c]`.

### Problem 1: Capacity Underuse

The optimal WGAN critic has **gradient norm exactly 1** almost everywhere (Proposition 1 in the paper). Weight clipping biases the network toward extremely simple functions that never use their full capacity.

```
Toy distributions — critic value surfaces:

  Weight Clipping          Gradient Penalty
  ┌─────────────┐          ┌─────────────┐
  │  ─────────  │          │  ~~~╮  ╭~~~ │
  │  ─────────  │          │   ╰╮╰──╯╭╯  │
  │  ─────────  │          │    ╰────╯   │
  │  ─────────  │          │  ╭──────╮   │
  └─────────────┘          └─────────────┘
  (flat/simple)            (captures detail)
```

Critics trained with weight clipping ignore higher moments of the data distribution — they only learn crude approximations.

### Problem 2: Exploding and Vanishing Gradients

The interaction between weight constraints and the loss function causes gradients to either:
- **Explode** (for large clipping threshold `c`)
- **Vanish** (for small clipping threshold `c`)

```
Gradient norm by layer (12-layer network on Swiss Roll):

                        ↑ Gradient norm (log scale)
   10 │
      │    ·
    0 │────────────────────────── Gradient Penalty (stable)
      │                    ·
  -10 │              ·
      │         ·
  -20 │    ·
      └─────────────────────────→ Layer (deep → shallow)
         Weight clipping (c=0.01): explodes or vanishes
```

### Problem 3: Bimodal Weight Distribution

Weight clipping pushes all weights toward the extremes of `[-c, c]`, forcing the network to act as a binary function rather than a nuanced one.

```
Weight clipping:          Gradient penalty:
   ↑ freq                    ↑ freq
   │ █             █         │
   │ █             █         │    ████████
   │ █             █         │   █████████
   └──────────────────→ w    └──────────────────→ w
    -c             +c          (smooth distribution)
```

---

## The Fix: Gradient Penalty

### Core Insight

A differentiable function is 1-Lipschitz **if and only if** its gradient norm is ≤ 1 everywhere. Instead of constraining weights, directly penalize the gradient norm.

### The Objective

```
L = E_{x̃~P_g}[D(x̃)]  -  E_{x~P_r}[D(x)]  +  λ · E_{x̂~P_x̂}[(‖∇_{x̂} D(x̂)‖₂ - 1)²]
    ─────────────────────────────────────────    ────────────────────────────────────────
                Original critic loss                         Gradient penalty
```

Where:
- `λ = 10` (penalty coefficient, works across all experiments)
- `x̂` is sampled uniformly along straight lines between real and generated samples

### Where to Sample x̂

```
Real sample x  ────────────────────────────  Fake sample x̃
              ε=0                        ε=1

               x̂ = ε·x + (1-ε)·x̃,   ε ~ U[0,1]
```

**Why straight lines?** Proposition 1 shows that the optimal critic has unit gradient norm along straight lines connecting paired real and fake samples. So enforcing the constraint on those lines is both necessary and sufficient.

### Two-Sided vs One-Sided Penalty

The paper uses a **two-sided penalty** that pushes the gradient norm toward exactly 1 (not just below 1). This works because the optimal critic already has gradient norm ≈ 1 everywhere relevant.

```
One-sided:   penalize if ‖∇D‖ > 1
Two-sided:   penalize if ‖∇D‖ ≠ 1   ← used in WGAN-GP
```

---

## Algorithm

```
Algorithm: WGAN-GP
───────────────────────────────────────────────────────
Hyperparameters: λ=10, n_critic=5, α=0.0001, β₁=0, β₂=0.9

while θ not converged:

    ╔══ Critic update loop (n_critic times) ════════════╗
    ║  for i = 1..m:                                    ║
    ║    Sample x ~ P_r            (real data)          ║
    ║    Sample z ~ p(z)           (noise)              ║
    ║    Sample ε ~ U[0,1]                              ║
    ║                                                   ║
    ║    x̃ ← G_θ(z)               (fake sample)        ║
    ║    x̂ ← ε·x + (1-ε)·x̃       (interpolated)       ║
    ║                                                   ║
    ║    L⁽ⁱ⁾ ← D_w(x̃) - D_w(x)                       ║
    ║         + λ·(‖∇_{x̂} D_w(x̂)‖₂ - 1)²             ║
    ║  end                                              ║
    ║                                                   ║
    ║  w ← Adam(∇_w mean(L), w, α, β₁, β₂)            ║
    ╚═══════════════════════════════════════════════════╝

    ╔══ Generator update ════════════════════════════════╗
    ║  Sample {z⁽ⁱ⁾} ~ p(z)                            ║
    ║  θ ← Adam(∇_θ mean(-D_w(G_θ(z))), θ, α, β₁, β₂) ║
    ╚═══════════════════════════════════════════════════╝
```

---

## Key Implementation Details

### 1. No Batch Normalization in the Critic

Batch norm changes the critic from mapping **one input → one output** to mapping **a batch → a batch**. This invalidates the per-input gradient penalty.

```python
# ✗ Don't do this in the critic:
x = nn.BatchNorm2d(channels)(x)

# ✓ Use layer norm instead:
x = nn.LayerNorm(normalized_shape)(x)
```

Layer normalization is a drop-in replacement that doesn't introduce batch-level correlations.

### 2. Computing the Gradient Penalty

```python
def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    
    # Random interpolation
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    
    # Critic score on interpolated samples
    score = critic(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Flatten and compute norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    
    # Penalty: (‖∇D‖ - 1)²
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty
```

### 3. Critic Training Loop

```python
LAMBDA = 10      # Gradient penalty coefficient
N_CRITIC = 5     # Critic steps per generator step

for real_batch in dataloader:
    for _ in range(N_CRITIC):
        noise = torch.randn(batch_size, latent_dim)
        fake = generator(noise).detach()
        
        critic_real = critic(real_batch).mean()
        critic_fake = critic(fake).mean()
        gp = gradient_penalty(critic, real_batch, fake, device)
        
        critic_loss = critic_fake - critic_real + LAMBDA * gp
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
    
    # Generator step
    noise = torch.randn(batch_size, latent_dim)
    fake = generator(noise)
    gen_loss = -critic(fake).mean()
    
    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()
```

---

## Results

### Convergence on CIFAR-10 (Inception Score)

```
Score
  7 ┤                    ···················· WGAN-GP (Adam)
    │                  ··
  6 ┤              ····   ···················· WGAN-GP (RMSProp) / DCGAN
    │           ···
  5 ┤       ····
    │    ···
  4 ┤  ···
    │··
  3 ┤ · Weight clipping (plateaus early)
    │
  2 ┤
    └──────────────────────────────────→ Generator iterations (×10⁴)

WGAN-GP significantly outperforms weight clipping.
```

### Inception Scores on CIFAR-10

| Method | Score (Unsupervised) |
|--------|---------------------|
| ALI | 5.34 ± .05 |
| BEGAN | 5.62 |
| DCGAN | 6.16 ± .07 |
| Improved GAN | 6.86 ± .06 |
| DFM | 7.72 ± .13 |
| **WGAN-GP ResNet (ours)** | **7.86 ± .07** |

| Method | Score (Supervised) |
|--------|-------------------|
| Improved GAN | 8.09 ± .07 |
| AC-GAN | 8.25 ± .07 |
| **WGAN-GP ResNet (ours)** | **8.42 ± .10** |
| SGAN | 8.59 ± .12 |

### Architecture Robustness (200 random architectures on ImageNet 32×32)

At inception score threshold ≥ 5.0:

```
           Only GAN succeeded:    0  ░░░░░░░░░░░░░░░░░░░░
         Only WGAN-GP succeeded: 147  ████████████████████████████
                  Both succeeded:  42  ████████
                     Both failed:  11  ██
```

WGAN-GP successfully trains architectures that standard GAN completely fails on, **including 101-layer ResNets** — believed to be the first time very deep residual networks were successfully trained in a GAN setting.

---

## Usage

### Quick Start

```bash
git clone https://github.com/your-username/your-repo
cd your-repo
pip install -r requirements.txt
python train.py --dataset cifar10 --lambda_gp 10 --n_critic 5
```

### Minimal Example

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            # ... upsample blocks
        )

class Critic(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # Note: LayerNorm, NOT BatchNorm
            nn.Conv2d(img_channels, 64, 4, 2, 1),
            nn.LayerNorm([64, 16, 16]),
            nn.LeakyReLU(0.2),
            # ...
        )
    
    def forward(self, x):
        return self.net(x)  # No sigmoid — output is unbounded
```

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lambda_gp` | 10 | Gradient penalty coefficient. Robust across all experiments. |
| `n_critic` | 5 | Critic iterations per generator iteration |
| `lr` | 0.0001 | Adam learning rate |
| `beta_1` | 0 | Adam β₁ (not 0.9 — using momentum destabilizes training) |
| `beta_2` | 0.9 | Adam β₂ |
| `batch_size` | 64 | — |

> **Note on β₁ = 0**: Using the default Adam β₁ = 0.9 can destabilize training. Setting β₁ = 0 eliminates momentum in the first moment estimate.

---

## Notes on Activation Functions

The gradient penalty involves second derivatives of the activation function. This causes issues with some activations:

| Activation | Works? | Notes |
|------------|--------|-------|
| ReLU | ✓ | Works empirically despite non-smooth gradient |
| Leaky ReLU | ✓ | Works well |
| tanh | ✓ | Works well |
| softplus | ✓ | Smooth, works well |
| ELU | ✗ | Derivative is continuous but not smooth — causes training failure |

---

## Citation

```bibtex
@article{gulrajani2017improved,
  title   = {Improved Training of Wasserstein GANs},
  author  = {Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and
             Dumoulin, Vincent and Courville, Aaron},
  journal = {arXiv preprint arXiv:1704.00028},
  year    = {2017}
}
```

Also cite the original WGAN paper:

```bibtex
@article{arjovsky2017wasserstein,
  title   = {Wasserstein GAN},
  author  = {Arjovsky, Martin and Chintala, Soumith and Bottou, L{\'e}on},
  journal = {arXiv preprint arXiv:1701.07875},
  year    = {2017}
}
```

---

## Further Reading

- [Original WGAN paper](https://arxiv.org/abs/1701.07875) — Arjovsky et al., 2017
- [Towards Principled Methods for Training GANs](https://arxiv.org/abs/1701.04862) — Arjovsky & Bottou, 2017
- [Official WGAN-GP implementation](https://github.com/igul222/improved_wgan_training)
