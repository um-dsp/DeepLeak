import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.segmentation import slic

def normalize_attributions(attributions):
    attributions -= attributions.min()
    attributions /= attributions.max()
    return attributions

def summarize_attributions(attributions):
    return np.sum(np.abs(attributions), axis=-1)

def plot_feature_importance_average(avg_attributions):
    plt.figure(figsize=(10, 10))
    if isinstance(avg_attributions, torch.Tensor):
        avg_attributions = avg_attributions.cpu().detach().numpy()
    avg_attributions = normalize_attributions(avg_attributions)
    summary = summarize_attributions(avg_attributions)
    plt.imshow(summary, cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.title("Average Feature Importance Heatmap")
    plt.show()

def create_feature_mask_from_topk(saliency_map, k=500):
    saliency_map = np.abs(saliency_map)
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
    flat = saliency_map.flatten()
    top_k = np.argpartition(flat, -k)[-k:]
    mask = np.zeros_like(flat)
    mask[top_k] = 1
    mask = mask.reshape(saliency_map.shape)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=0)
    return torch.tensor(mask)

def create_feature_masks_captum(batch_imgs, n_segments=15, compactness=20, **kwargs):
    masks = []
    for i in range(batch_imgs.shape[0]):
        img_np = batch_imgs[i].squeeze().permute(1, 2, 0).cpu().detach().numpy()
        segments = slic(img_np, n_segments=n_segments, compactness=compactness)
        feature_mask = torch.tensor(segments, dtype=torch.long).unsqueeze(0)
        masks.append(feature_mask)
    return torch.stack(masks).to(batch_imgs.device)

def get_model_predictions(model, batch_imgs):
    model.eval()
    with torch.no_grad():
        output = model(batch_imgs)
        return torch.argmax(output, dim=1)

def get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found.")
    return last_conv

def perturb_fn(inputs):
    noise = torch.randn_like(inputs) * 0.1
    return noise, inputs - noise

def perturb_fns(inputs):
    return torch.randn_like(inputs) * 0.1
