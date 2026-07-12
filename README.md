# ADAPT-VR: Proof-of-Concept Implementation

**Conditional GAN-Based Scene Parameter Control for Adaptive Virtual Reality Exposure Therapy**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.21326066-blue)](https://doi.org/10.5281/zenodo.21326066)

> **Author:** Dr Samuel Duraivel  
> **Affiliation:** Jubilee Mission Group of Institutions, Bangalore, India  
> **ORCID:** [0000-0002-5287-2500](https://orcid.org/0000-0002-5287-2500)  
> **Patent:** System and Method for Adaptive VRET Using GANs (provisional, 2024)  
> **Technical Report:** [Zenodo DOI: 10.5281/zenodo.21326066](https://doi.org/10.5281/zenodo.21326066)  
> **Conceptual Preprint:** [Preprints.org DOI: 10.20944/preprints202409.0107.v1](https://doi.org/10.20944/preprints202409.0107.v1)

---

## Overview

This repository contains the proof-of-concept implementation of a **conditional Generative Adversarial Network (cGAN)** that generates virtual environment scene parameters conditioned on a composite biometric anxiety index. It is the technical foundation for the **ADAPT-VR** research programme (HORIZON-MSCA-2026-PF-01).

The system maps a real-time anxiety signal α ∈ [0,1] — derived from heart rate (HR) and galvanic skin response (GSR) — to five scene parameters:

| Parameter | Description |
|---|---|
| Crowd Density | Density of avatars in the scene |
| Spatial Proximity | Distance of avatars to the patient |
| Ambient Sound | Background noise level |
| Lighting Intensity | Scene brightness and harshness |
| Avatar Motion | Speed of avatar movement |

The mapping follows **graduated exposure therapy principles** (Craske et al., 2014): stimuli increase through mid-anxiety ranges and automatically reduce when anxiety exceeds a therapeutic safety threshold.

---

## Key Results

| Parameter | Pearson r | MAE | Diversity (SD) |
|---|---|---|---|
| Crowd Density | 0.972 | 0.050 | 0.023 |
| Spatial Proximity | 0.982 | 0.017 | 0.028 |
| Ambient Sound | 0.990 | 0.030 | 0.025 |
| Lighting Intensity | 0.727 | 0.043 | 0.037 |
| Avatar Motion | 0.994 | 0.027 | 0.030 |
| **Mean** | **0.933** | **0.033** | **0.029** |

**Safety protocol:** Crowd density peaks at α = 0.72 (target: 0.65) — the system activates the safety mechanism conservatively early, consistent with therapeutic best practice.

---

## Installation

```bash
git clone https://github.com/duraivelsamuel/ADAPT-VR-PoC.git
cd ADAPT-VR-PoC
pip install -r requirements.txt
```

**No GPU required.** Trains in ~10 minutes on CPU, ~2 minutes on Google Colab GPU.

---

## Usage

```bash
python adapt_vr_poc.py
```

All outputs are saved to `adapt_vr_poc_results/`:

```
adapt_vr_poc_results/
├── training_loss.png              # cGAN training convergence
├── figure1_parameter_response.png # Core result: parameter curves vs targets
├── figure2_scene_grid.png         # Visual scene representation per anxiety stage
├── figure3_biometric_conditioning.png  # End-to-end pipeline for 5 patients
├── adapt_vr_poc_model.pt          # Saved model weights
└── results_summary.txt            # Quantitative evaluation report
```

### Run on Google Colab (free GPU)

```python
!git clone https://github.com/duraivelsamuel/ADAPT-VR-PoC.git
%cd ADAPT-VR-PoC
!pip install -r requirements.txt
!python adapt_vr_poc.py
```

---

## Architecture

```
Input: z (32-dim noise) + α (scalar anxiety index)
         ↓
    Generator G(z, α)
    FC(34→128) → LayerNorm → LeakyReLU
    FC(128→128) → LayerNorm → LeakyReLU  
    FC(128→64) → LeakyReLU
    FC(64→5) → Sigmoid
         ↓
    Scene parameters p ∈ [0,1]^5

    Discriminator D(p, α)
    FC(6→128) → LeakyReLU → Dropout(0.3)
    FC(128→128) → LeakyReLU → Dropout(0.3)
    FC(128→64) → LeakyReLU
    FC(64→1) → Sigmoid
```

**Training objective:**
```
L_G = L_adv + 10·L_recon + 8·L_safety

L_safety = ReLU(P1 - 0.65) + ReLU(P3 - 0.70)   [applied when α > 0.75]
```

---

## Reproducibility

All results are fully reproducible:
```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
```

No external datasets required. Synthetic biometric data is generated from clinically validated reference ranges (Boucsein, 2012; Critchley, 2002).

---

## Clinical Context

This implementation is a proof-of-concept for the **ADAPT-VR** research programme, which will:

- **WP1:** Co-design workshops with NHS clinicians and SAD service users to validate the therapeutic parameter mapping
- **WP2:** Integrate the cGAN with a photorealistic VR rendering pipeline (Meta Quest 3, HoloLens 2, BCU HCI Research Centre)
- **WP3:** Pilot randomised controlled trial (n=60, adaptive GAN-VRET vs. static-VRET, NHS ethics approved)

**Note:** This proof-of-concept uses synthetic biometric data. Clinical deployment requires ethics approval, real biometric validation, and photorealistic VR integration. This code is not a medical device.

---

## Citation

If you use this code or the associated technical report, please cite:

```bibtex
@misc{duraivel2026adaptvrpoc,
  title     = {Conditional {GAN}-Based Scene Parameter Control for Adaptive 
               Virtual Reality Exposure Therapy: A Proof-of-Concept Implementation},
  author    = {Duraivel, Samuel},
  year      = {2026},
  doi       = {10.5281/zenodo.21326066},
  url       = {https://doi.org/10.5281/zenodo.21326066},
  publisher = {Zenodo}
}
```

---

## Related Work

- **Technical report (this implementation):** Duraivel, S. (2026). Conditional GAN-Based Scene Parameter Control for Adaptive Virtual Reality Exposure Therapy: A Proof-of-Concept Implementation. *Zenodo*. DOI: [10.5281/zenodo.21326066](https://doi.org/10.5281/zenodo.21326066)
- **Preprint (conceptual framework):** Duraivel, S. (2024). Enhancing virtual reality exposure therapy for social anxiety disorder using generative adversarial networks. *Preprints.org*. DOI: [10.20944/preprints202409.0107.v1](https://doi.org/10.20944/preprints202409.0107.v1)
- **Q1 Publication (socio-technical XR framework):** Vasudevan, S., Piazza, A., Rajendran, L., & Duraivel, S. (2026). Mapping the Metaverse Minefield: A TIPS Framework. *Computers & Security*, 160, 104710.
- **Q1 Publication (computational XR discourse):** Duraivel, S., et al. (2026). Public Sentiment and Thematic Evolution in the Metaverse. *PLOS ONE*, 21(5), e0345135.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## Contact

Dr Samuel Duraivel | duraivelsamuel@gmail.com | [ORCID](https://orcid.org/0000-0002-5287-2500)
