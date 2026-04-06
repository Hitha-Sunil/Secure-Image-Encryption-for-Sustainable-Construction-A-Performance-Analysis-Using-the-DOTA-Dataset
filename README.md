# Secure-Image-Encryption-for-Sustainable-Construction-A-Performance-Analysis-Using-the-DOTA-Dataset
A Comparative Study using Autoencoder, Chaos, Quantum-Inspired, and Transformer-Based Models
📌 Overview

This project presents a comparative analysis of four image encryption–decryption techniques applied to aerial imagery from the DOTA dataset. The goal is to evaluate trade-offs between:

🔒 Encryption strength
🖼️ Reconstruction fidelity
🤖 AI compatibility (classification performance)

The four approaches studied:

Autoencoder-Based Encryption
Chaos-Based Encryption (Henon + Logistic Map)
Quantum-Inspired Encryption
Transformer-Based Encryption (Vision Transformer-inspired)
| Metric           | Autoencoder | Chaos  | Transformer | Quantum |
| ---------------- | ----------- | ------ | ----------- | ------- |
| **SSIM**         | 0.9952      | 0.9878 | 0.8077      | 0.9800  |
| **NPCR (%)**     | 99.67       | 99.41  | 100.00      | 77.37   |
| **UACI (%)**     | 29.31       | 26.27  | 3.01        | 26.00   |
| **Accuracy (%)** | 97.15       | 91.18  | 98.04       | 98.00   |

🧠 Methodology Summary
🔹 Autoencoder-Based Method
Fully connected encoder-decoder architecture
Noise-based encryption using key-seeded perturbation
Optimized using MSE loss
🔹 Chaos-Based Method
Permutation: Henon Map
Diffusion: Logistic Map
Strong statistical security (high entropy)
🔹 Quantum-Inspired Method
XOR-based transformation with quantum-style keying
Focus on reversible secure mapping
🔹 Transformer-Based Method
Patch embedding (16×16 patches)
Multi-head self-attention
Learns implicit encryption via feature transformation

📈 Evaluation Metrics

The models are evaluated using:

SSIM – Structural similarity
NPCR – Pixel change rate
UACI – Intensity variation
Entropy – Randomness
Classification Accuracy
PCA & t-SNE – Feature preservation
📊 Visualizations

The project includes:

SSIM Distribution Plots
Entropy Histograms
PCA Feature Projections
t-SNE Embeddings

These help analyze both:

Reconstruction quality
Feature preservation
