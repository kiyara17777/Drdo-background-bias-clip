# Analyzing the Effect of Background Change on Vision Model Predictions

**DRDO Research Internship Project** — DRDO SAG (Scientific Analysis Group)

**Student Intern:** Kiyara Chandrawat
**Mentor:** Ms. Mehak
**Duration:** 4 weeks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-56mokQzVRQELwURFhJclB6oJrw2pzqf)

---

## Abstract

Modern vision models often rely on both foreground objects and surrounding background context to interpret images. This project investigates how changes in image backgrounds affect the semantic predictions of a vision-language model.

A three-stage pipeline was implemented:
1. **Automated foreground extraction and background separation**
2. **Background feature embedding and similarity analysis** using cosine similarity
3. **Foreground compositing onto visually dissimilar backgrounds**, followed by semantic evaluation with the CLIP (ViT-B/32) model

Results show that background context can significantly influence model confidence and predicted labels. Foreground objects such as humans tend to dominate predictions, but object-centric scenes (e.g., vehicles, animals, beds) show measurable shifts in semantic interpretation after background replacement. Limitations in compositing realism introduced visual artifacts that may have constrained stronger conclusions.

---

## Tech Stack

| Component | Tool/Library |
|---|---|
| Deep Learning Framework | PyTorch |
| Vision-Language Model | CLIP (ViT-B/32) |
| Feature Extraction | ResNet-18 |
| Image Processing | OpenCV |
| Foreground Segmentation | rembg (U²-Net) |
| Similarity Metric | Cosine Similarity |

**Dataset:** [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) (streamed subset, ~8,727 filtered images)

---

## Project Workflow

### Week 1 — Data Preparation
- Streamed images directly from LAION-400M URLs (no full dataset download)
- Extracted foreground objects using **rembg** (U²-Net based salient object segmentation)
- Generated binary masks, foreground objects (transparent background), and isolated background images
- Filtered dataset by resolution, foreground area, and mask quality → **8,727 high-quality images** retained

### Week 2 — Background Similarity Analysis
- Converted each background into a feature embedding using pretrained **ResNet-18**
- Computed an 8,727 × 8,727 **cosine similarity matrix** across all backgrounds
- Identified visually dissimilar background pairs for controlled replacement experiments

### Week 3 — Background Replacement & Semantic Evaluation
- Composited foreground objects onto dissimilar backgrounds using **alpha blending**
- Evaluated the modified images with **CLIP (ViT-B/32)** against a label set: `person, car, bed, animal, vehicle, room, indoor, outdoor`
- Analyzed dominant label distribution, average confidence per label, and a **confidence gap metric** to quantify background sensitivity

---

## Pipeline Overview

```
Original Image
      │
      ▼
Foreground Extraction (rembg / U²-Net)
      │
      ├──► Foreground Object (transparent bg)
      └──► Background Image
                  │
                  ▼
        ResNet-18 Feature Embedding
                  │
                  ▼
        Cosine Similarity Matrix
                  │
                  ▼
     Select Visually Dissimilar Background
                  │
                  ▼
     Alpha Blending (Foreground + New Background)
                  │
                  ▼
        CLIP (ViT-B/32) Semantic Evaluation
                  │
                  ▼
    Label Predictions + Confidence Scores
```

---

## Key Results

- **Dominant label distribution:** `person` was the most frequent top-predicted label after background change, followed by `vehicle` and `animal`; `indoor` appeared least often.
- **Average confidence per label:** `person` had the highest mean confidence; environment-only labels like `outdoor` had the weakest confidence, suggesting background alone provides limited semantic signal.
- **Background sensitivity:** Object-centric scenes (cars, animals, beds) showed the largest confidence gaps after background swaps, while human-centric images were comparatively robust to background change — indicating strong foreground dominance in CLIP's predictions.

See the full [Project Report](./report/DRDO_Project_Report.pdf) for detailed figures, methodology, and analysis.

---

## Limitations

- Imperfect foreground segmentation (residual background in masks)
- Compositing artifacts: edge halos, lighting/color mismatch, scale and depth inconsistencies
- Limited label set for semantic evaluation
- No depth-aware or gradient-domain (Poisson) blending used

---

## Future Work

- Depth-aware compositing for more realistic scene integration
- Poisson / gradient-domain blending for seamless foreground-background merging
- Diffusion-based background generation for greater realism and diversity
- Cross-architecture testing beyond CLIP to assess contextual bias more broadly
- Expanded label set for richer semantic evaluation

---

## Repository Structure

```
├── README.md
├── DRDO_INTERNSHIP.ipynb        # Main Colab notebook
├── report/
│   └── DRDO_Project_Report.pdf  # Full project report
├── results/
│   └── sample_outputs/          # Example figures (masks, composites, plots)
└── requirements.txt
```

---

## Running the Notebook

The notebook streams data directly from LAION-400M, so no local dataset download is required.

1. Open the notebook: [DRDO_INTERNSHIP.ipynb on Colab](https://colab.research.google.com/drive/1-56mokQzVRQELwURFhJclB6oJrw2pzqf)
2. Run all cells in order (Runtime → Run all)
3. GPU runtime recommended (Runtime → Change runtime type → GPU) for faster CLIP/ResNet inference

Or clone this repo and run locally:

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
jupyter notebook DRDO_INTERNSHIP.ipynb
```

---

## References

1. Radford, A. et al. — *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*
2. Schuhmann, C. et al. — *LAION-400M Dataset*
3. He, K. et al. — *Deep Residual Learning for Image Recognition (ResNet)*
4. Qin, X. et al. — *U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection*

---

*This project was completed as part of a research internship at DRDO – SAG.*
