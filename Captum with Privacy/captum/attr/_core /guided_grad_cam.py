#!/usr/bin/env python3
import warnings
from typing import Any, List, Union
import torch
import numpy as np
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.guided_backprop_deconvnet import GuidedBackprop
from captum.attr._core.layer.grad_cam import LayerGradCam
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class GuidedGradCam(GradientAttribution):
    """
    Computes element-wise product of guided backpropagation attributions
    with upsampled (non-negative) GradCAM attributions.

    Now includes minor privacy safeguards.
    """

    def __init__(
        self, model: Module, layer: Module, device_ids: Union[None, List[int]] = None
    ) -> None:
        """
        Args:
            model (nn.Module): The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which GradCAM attributions are computed.
            device_ids (list[int]): Device ID list for DataParallel models.
        """
        super().__init__(model)
        self.grad_cam = LayerGradCam(model, layer, device_ids)
        self.guided_backprop = GuidedBackprop(model)

        # Store privacy parameters for consistency
        self.correc = True
        self.noise_level = 0.01
        self.clamp_range = (-1, 1)
        self.precision = 4
        self.mask_threshold = 1e-3

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        interpolate_mode: str = "nearest",
        attribute_to_layer_input: bool = False,
        correc: bool = True,
        noise_level: float = 0.01,  # Noise added to attributions
        clamp_range: tuple = (-1, 1),  # Limits extreme values
        precision: int = 4,  # Rounds to limit precision
        mask_threshold: float = 1e-3,  # Zeroes out small attributions
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Computes GuidedGradCAM attributions with privacy protection.
        """
        # Store parameters as instance variables
        self.correc = correc
        self.noise_level = noise_level
        self.clamp_range = clamp_range
        self.precision = precision
        self.mask_threshold = mask_threshold

        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_tensor_into_tuples(inputs)

        # Compute GradCAM attributions
        grad_cam_attr = self.grad_cam.attribute.__wrapped__(
            self.grad_cam,
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            relu_attributions=True,
            correc = False

        )
        if isinstance(grad_cam_attr, tuple):
            assert len(grad_cam_attr) == 1, (
                "GuidedGradCAM attributions for layer with multiple inputs / "
                "outputs is not supported."
            )
            grad_cam_attr = grad_cam_attr[0]

        # Compute Guided Backpropagation attributions
        guided_backprop_attr = self.guided_backprop.attribute.__wrapped__(
            self.guided_backprop,
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
        )

        # Process attributions with privacy transformations
        output_attr: List[Tensor] = []
        for i in range(len(inputs)):
            try:
                combined_attr = guided_backprop_attr[i] * LayerAttribution.interpolate(
                    grad_cam_attr,
                    inputs[i].shape[2:],
                    interpolate_mode=interpolate_mode,
                )
                if self.correc:
                    output_attr.append(self._process_attribution(combined_attr))
                else:
                    output_attr.append(combined_attr)
            except Exception:
                warnings.warn(
                    "Couldn't appropriately interpolate GradCAM attributions for some "
                    "input tensors, returning empty tensor for corresponding "
                    "attributions."
                )
                output_attr.append(torch.empty(0))

        return _format_output(is_inputs_tuple, tuple(output_attr))

    def _process_attribution(self, attributions: Tensor) -> Tensor:
        """
        Applies privacy transformations to attributions.
        """
        noise = torch.randn_like(attributions) * self.noise_level
        attributions = attributions + noise  # Add noise
        attributions = torch.clamp(attributions, *self.clamp_range)  # Clamp values
        attributions = torch.round(attributions * (10**self.precision)) / (10**self.precision)  # Reduce precision
        attributions[torch.abs(attributions) < self.mask_threshold] = 0  # Mask small values
        return attributions
