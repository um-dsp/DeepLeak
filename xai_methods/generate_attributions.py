import os
import torch
import pickle
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset

from captum.metrics import infidelity, sensitivity_max

from datasets import DynamicGraphDataset
from utils.helpers import perturb_fn, perturb_fns, create_feature_masks_captum

from .captum_wrappers import *
from .attribution_wrappers import AnchorsAttribution, ProtoDashAttribution


def dispatch_xai_method(xai, model, model_ini, batch_imgs, batch_labels, train_data, kwargs):
    kwargss = {"n_samples": kwargs["n_samples"]} if "n_samples" in kwargs else {}

    if xai == "SMAP":
        return Saliency_map(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "GBackProp":
        return GuidedBackprop_attri(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "IG":
        return integrated_gradients(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "SHAP":
        feature_mask = create_feature_masks_captum(batch_imgs, **kwargs)
        return ShapleyValues_attri(model, batch_imgs, target=batch_labels, feature_mask=feature_mask, **kwargss)
    elif xai == "LIME":
        feature_mask = create_feature_masks_captum(batch_imgs, **kwargs)
        return Lime_attrib(model, batch_imgs, feature_mask, **kwargss)
    elif xai == "LRP":
        return LRP_attri(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "SmoothGrad":
        return SmoothGrad_attri(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "VarGrad":
        return VarGrad_attri(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "DeepLift":
        return DeepLift_attri(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "Occlusion":
        return Occlusion_attri(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "GGC":
        return GGC_attri(model, model_ini, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "GC":
        return GC_attri(model, model_ini, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "KSHAP":
        feature_mask = create_feature_masks_captum(batch_imgs)
        return Kernel_shap_attri(model, batch_imgs, target=batch_labels, feature_mask=feature_mask, **kwargs)
    elif xai == "DCAttr":
        return deconv_attri(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "INGRAttr":
        return input_x_grad_attri(model, batch_imgs, target=batch_labels, **kwargs)
    elif xai == "FeaAbAttr":
        feature_mask = create_feature_masks_captum(batch_imgs)
        return feature_ablation_attri(model, batch_imgs, target=batch_labels, feature_mask=feature_mask, **kwargs)
    elif xai == "FeaPermAttr":
        feature_mask = create_feature_masks_captum(batch_imgs)
        return feature_perm_attri(model, batch_imgs, batch_labels, feature_mask=feature_mask, **kwargs)
    elif xai == "ProtoDa":
        return ProtoDashAttribution(model).attribute(batch_imgs, target=batch_labels, train_data=train_data), None, None
    elif xai == "Anchor":
        return AnchorsAttribution(model).attribute(batch_imgs, target=batch_labels), None, None
    else:
        raise ValueError(f"XAI method '{xai}' not supported.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gen_attri(save_path, subset, model_ini, xai, target=0, batch_size=252, metrics=False, **kwargs):
    dataset = DynamicGraphDataset(root=save_path)
    model = torch.nn.DataParallel(model_ini).to(next(model_ini.parameters()).to(device))

    img_tensors = [subset[i][0].unsqueeze(0).to(device) for i in range(len(subset))]
    labels = [subset[i][1] for i in range(len(subset))]
    train_data = torch.cat(img_tensors).cpu().numpy()

    tensor_dataset = TensorDataset(torch.cat(img_tensors), torch.tensor(labels))
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    sensitivity_scores, infidelity_scores = [], []

    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs, batch_labels = batch_imgs.to(device), batch_labels.to(device)

        try:
            attributions, method, attributions_sc = dispatch_xai_method(
                xai, model, model_ini, batch_imgs, batch_labels, train_data, kwargs
            )
        except Exception as e:
            print(f"[Error] XAI: {xai}, Batch skipped. {str(e)}")
            continue

        # Infidelity
        with torch.no_grad():
            inf_score = infidelity(model, perturb_fn, batch_imgs, attributions_sc, target=batch_labels)

        # Store graph dataset for this batch
        for i, attr in enumerate(attributions):
            data = Data(x=torch.tensor(attr, dtype=torch.float))
            if i < len(batch_labels):
                dataset.add_attribut(data, float(target), float(batch_labels[i]))

        # Sensitivity
        if metrics and method is not None:
            with torch.no_grad():
                feature_mask = kwargs.get("feature_mask", None)
                if xai in ["Occlusion", "SHAP", "LIME", "FeaPermAttr", "FeaAbAttr", "KSHAP", "ProtoDa"]:
                    sens_score = sensitivity_max(method.attribute, batch_imgs, perturb_func=perturb_fns,
                                                 target=batch_labels, feature_mask=feature_mask, n_perturb_samples=1, **kwargs)
                else:
                    sens_score = sensitivity_max(method.attribute, batch_imgs, perturb_func=perturb_fns,
                                                 target=batch_labels, n_perturb_samples=1, **kwargs)

            sensitivity_scores.append(sens_score.cpu())
            infidelity_scores.append(inf_score)

    dataset.save()

    if metrics:
        sens_all = torch.cat(sensitivity_scores)
        infid_all = torch.cat(infidelity_scores)
        with open(f"{save_path}_sensitivity_scores.pickle", "wb") as f:
            pickle.dump(sens_all, f)
        with open(f"{save_path}_infidelity_scores.pickle", "wb") as f:
            pickle.dump(infid_all, f)
        return sens_all.mean().item()

    return 0
