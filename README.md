# DeepLeak: Privacy Enhancing Hardening of Model Explanations Against Membership Leakage

This repository containts code to reproduce results for the paper LeakyFarm on CIFAR-100.

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
XAI methods keywords SMAP: Saliency Map, GBackProp: Guided BackProp, IG: Integrated Gradients, SHAP: SHAP, LIME: LIMESmoothGrad: SmoothGrad, VarGrad: VarGrad, DeepLift: DeepLIFT, Occlusion: Occlusion, GGC: GradCam++, GC:GradCAM, KSHAP: K, DCAttr: Deconvolution, INGRAttr: InputXGrad, ProtoDa: ProtoDash, Anchor: Anchors

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
