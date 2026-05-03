# TUS-GAN: Time-Use Survey GAN
 
## Overview
 
TUS-GAN is a research project that reframes the problem of **incomplete time-use diaries** as an **image inpainting task**, leveraging generative adversarial networks to impute missing survey entries.
 
## Motivation
 
Time-use surveys track how individuals allocate time across activities throughout a day. In practice, respondents often leave entries incomplete — creating gaps that undermine downstream analysis. Traditional statistical imputation methods fail to capture the rich sequential and contextual structure of these diaries. TUS-GAN addresses this by treating missing entries as "holes" in a structured image, enabling spatially-aware deep learning models to fill them.
 
## Approach
 
The diary is encoded as a 2D representation where rows correspond to time slots and columns to activity categories. Missing entries appear as masked regions — analogous to corrupted pixels in an image inpainting problem.
 
Two GAN architectures are employed:
 
- **WGAN-GP** — Wasserstein GAN with Gradient Penalty for training stability and improved convergence
- **Pix2Pix** — Conditional GAN for image-to-image translation, conditioning on the observed diary structure to generate plausible completions
## Key Contributions
 
- Novel framing of survey imputation as a visual completion problem
- Exploits sequential and categorical structure of time-use data via image encoding
- GAN-based generation preserves realistic activity patterns and temporal coherence
## Stack
 
| Component | Detail |
|---|---|
| Framework | PyTorch |
| Models | WGAN-GP, Pix2Pix |
| Domain | Time-use survey data |
| Lab | SCULPT-Lab |
| Supervisor | Dr. Agnivesh Pani |
 
## Status
 
Active research internship (May – July 2026).
