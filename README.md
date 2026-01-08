# DeepLeak: Privacy Enhancing Hardening of Model Explanations Against Membership Leakage
This repository contains the implementation of our IEEE SaTML 2026 paper **[DeepLeak: Privacy Enhancing Hardening of Model Explanations Against Membership Leakage](https://arxiv.org/pdf/2601.03429)**. DeepLeak is a system to audit and mitigate privacy risks in post-hoc explanation methods. It advances the state-of-the-art in three ways: (1) comprehensive leakage profiling: we develop a stronger explanation-aware membership inference attack to quantify how much representative explanation methods leak membership information under default configurations; (2) lightweight hardening strategies: we introduce practical, model-agnostic mitigations, including sensitivity-calibrated noise, attribution clipping, and masking, that substantially reduce membership leakage while preserving explanation utility; and (3) root-cause analysis: through controlled experiments, we pinpoint algorithmic properties that drive leakage in ML explanation methods. 

This repository containts code to reproduce results for the paper on CIFAR-100.

* **Attack Phase**: Select top-k seeds based on the most vulnerable settings per XAI method.
* **Optimization Phase**: Explore parameter configurations that minimize privacy leakage while preserving explanation utility.

---

## Directory Structure

```
DeepLeak/
│
├── data/                     # Dataset pickle files and attribution outputs
├── datasets/                 # Dataset loading and splitting utilities
├── models/                   # Model definitions and pretrained weights
├── utils/                    # Training and evaluation utilities
├── xai_methods/              # Attribution generators and wrappers
│   └── captum_wrappers.py    # Captum-specific method modifications
│   └── attribution_wrappers.py
│   └── generate_attributions.py
```

---

## How to Run

### Step 1: Set Up Environment

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

### Step 2: Run the Pipeline
XAI methods keywords:  SMAP: Saliency Map, GBackProp: Guided BackProp, IG: Integrated Gradients, SHAP: SHAP, LIME: LIMESmoothGrad: SmoothGrad, VarGrad: VarGrad, DeepLift: DeepLIFT, Occlusion: Occlusion, GGC: GradCam++, GC:GradCAM, KSHAP: K, DCAttr: Deconvolution, INGRAttr: InputXGrad, ProtoDa: ProtoDash, Anchor: Anchors

#### Attack Phase Only

```bash
python main.py --mode attack --xai_methods SMAP SHAP LIME --topk 3 --trials 20
```

#### Optimization Phase Only

```bash
python main.py --mode optimize
```

#### Run Both Phases

```bash
python main.py --mode both --xai_methods SMAP SHAP LIME --topk 3 --trials 20
```

---

## Notes on Captum XAI Methods

For XAI methods based on **Captum** (guided_backprop_deconvnet, guided_grad_cam.py, input_x_gradient, saliency), please make sure to modify your captum library with these privacy applied mechanisms. 

## Output

* Attack phase results will be saved to `top_3_tprs.csv`, listing the top seeds and their respective TPR scores.
* Optimized parameter settings and performance metrics are saved as `optuna_results_<XAI>+<seed>.pkl`.

---
