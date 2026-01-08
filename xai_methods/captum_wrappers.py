import torch
import numpy as np
from captum.attr import (
    IntegratedGradients, Saliency, GuidedBackprop, DeepLift, LRP,
    NoiseTunnel, Lime, FeatureAblation, FeaturePermutation, InputXGradient,
    KernelShap, ShapleyValueSampling, GuidedGradCam, Occlusion
)
from captum.attr._core.layer.grad_cam import LayerGradCam
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLinearRegression

from utils.helpers import get_model_predictions, get_last_conv_layer


def to_numpy(attributions_sc):
    return attributions_sc.squeeze().cpu().detach().numpy()


def integrated_gradients(model, img_tensor, target, **kwargs):
    ig = IntegratedGradients(model)
    attributions_sc, _ = ig.attribute(img_tensor, target=target, return_convergence_delta=True, **kwargs)
    return to_numpy(attributions_sc), ig, attributions_sc


def Saliency_map(model, img_tensor, target, **kwargs):
    ig = Saliency(model)
    attributions_sc = ig.attribute(img_tensor, target=target, **kwargs)
    return to_numpy(attributions_sc), ig, attributions_sc


def GuidedBackprop_attri(model, img_tensor, target, **kwargs):
    ig = GuidedBackprop(model)
    attributions_sc = ig.attribute(img_tensor, target=target, **kwargs)
    return to_numpy(attributions_sc), ig, attributions_sc


def DeepLift_attri(model, img_tensor, target, **kwargs):
    ig = DeepLift(model)
    attributions_sc = ig.attribute(img_tensor, target=target)
    return to_numpy(attributions_sc), ig, attributions_sc


def LRP_attri(model, img_tensor, target, **kwargs):
    ig = LRP(model)
    attributions_sc = ig.attribute(img_tensor, target=target)
    return to_numpy(attributions_sc), ig, attributions_sc


def SmoothGrad_attri(model, img_tensor, target, **kwargs):
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attributions_sc = nt.attribute(img_tensor, nt_type='smoothgrad', target=target, **kwargs)
    return to_numpy(attributions_sc), nt, attributions_sc


def VarGrad_attri(model, img_tensor, target, **kwargs):
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attributions_sc = nt.attribute(img_tensor, nt_type='vargrad', target=target, **kwargs)
    return to_numpy(attributions_sc), nt, attributions_sc


def Occlusion_attri(model, img_tensor, target, **kwargs):
    ig = Occlusion(model)
    attributions_sc = ig.attribute(img_tensor, sliding_window_shapes=(3, 16, 16),
                                   strides=(3, 1, 1), target=target, perturbations_per_eval=5)
    return to_numpy(attributions_sc), ig, attributions_sc


def Lime_attrib(model, batch_imgs, feature_masks, **kwargs):
    sim_fn = get_exp_kernel_similarity_function('euclidean', kernel_width=1000)
    lime = Lime(model, interpretable_model=SkLearnLinearRegression(), similarity_func=sim_fn)
    targets = get_model_predictions(model, batch_imgs)
    attributions_sc = lime.attribute(batch_imgs, target=targets, feature_mask=feature_masks,
                                     n_samples=kwargs.get("n_samples", 25))
    return attributions_sc.cpu().detach().numpy(), lime, attributions_sc


def Kernel_shap_attri(model, img_tensor, target=0, feature_mask=None, **kwargs):
    ig = KernelShap(model)
    attributions_sc = ig.attribute(img_tensor, target=target, feature_mask=feature_mask)
    return to_numpy(attributions_sc), ig, attributions_sc


def ShapleyValues_attri(model, img_tensor, target=0, feature_mask=None, **kwargs):
    svs = ShapleyValueSampling(model)
    targets = get_model_predictions(model, img_tensor)
    attributions_sc = svs.attribute(img_tensor, target=targets, feature_mask=feature_mask,
                                    n_samples=kwargs.get("n_samples", 25))
    return to_numpy(attributions_sc), svs, attributions_sc


def feature_ablation_attri(model, img_tensor, target, feature_mask, **kwargs):
    fa = FeatureAblation(model)
    attributions_sc = fa.attribute(img_tensor, target=target, feature_mask=feature_mask, **kwargs)
    return to_numpy(attributions_sc), fa, attributions_sc


def feature_perm_attri(model, img_tensor, target, feature_mask, **kwargs):
    fp = FeaturePermutation(model)
    attributions_sc = fp.attribute(img_tensor, target=target, feature_mask=feature_mask,
                                   perturbations_per_eval=16, **kwargs)
    return to_numpy(attributions_sc), fp, attributions_sc


def input_x_grad_attri(model, img_tensor, target, **kwargs):
    ixg = InputXGradient(model)
    attributions_sc = ixg.attribute(img_tensor, target=target, **kwargs)
    return to_numpy(attributions_sc), ixg, attributions_sc


def deconv_attri(model, img_tensor, target, **kwargs):
    dc = GuidedBackprop(model)  # Alternatively: use Deconvolution directly
    attributions_sc = dc.attribute(img_tensor, target=target, **kwargs)
    return to_numpy(attributions_sc), dc, attributions_sc


def GC_attri(model, model_ini, img_tensor, target, **kwargs):
    layer = get_last_conv_layer(model)
    ig = LayerGradCam(model, layer)
    attributions_sc = ig.attribute(img_tensor, target=target, upsample_to_input=True, **kwargs)
    return to_numpy(attributions_sc), ig, attributions_sc


def GGC_attri(model, model_ini, img_tensor, target, **kwargs):
    layer = get_last_conv_layer(model)
    ig = GuidedGradCam(model, layer)
    attributions_sc = ig.attribute(img_tensor, target=target, **kwargs)
    return to_numpy(attributions_sc), ig, attributions_sc
