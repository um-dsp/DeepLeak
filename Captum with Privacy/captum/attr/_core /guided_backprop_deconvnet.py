#!/usr/bin/env python3
import warnings
from typing import Any, List, Tuple, Union
import torch
import torch.nn.functional as F
from captum._utils.common import (
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
    _register_backward_hook,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


class ModifiedReluGradientAttribution(GradientAttribution):
    def __init__(self, model: Module, use_relu_grad_output: bool = False) -> None:
        """
        Args:
            model (nn.Module): The reference to PyTorch model instance.
        """
        super().__init__(model)
        self.model = model
        self.backward_hooks: List[RemovableHandle] = []
        self.use_relu_grad_output = use_relu_grad_output

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        correc: bool = True,
        noise_level: float = 0.01,  # Noise added to attributions
        clamp_range: tuple = (-1, 1),  # Limits extreme values
        precision: int = 4,  # Rounds to limit precision
        mask_threshold: float = 1e-3,  # Zeroes out small attributions
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Computes attribution by overriding ReLU gradients.
        Adds privacy protection through noise, clamping, and masking.
        """

        # Debugging to verify correct parameter values are always passed

        is_inputs_tuple = _is_tuple(inputs)
        inputs_tuple = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs_tuple)

        # Set hooks for overriding ReLU gradients
        warnings.warn(
            "Setting backward hooks on ReLU activations. The hooks will be removed after the attribution is finished."
        )

        try:
            self.model.apply(self._register_hooks)
            gradients = self.gradient_func(
                self.forward_func, inputs_tuple, target, additional_forward_args
            )

            if correc:
                attributions = tuple(
                    self._process_attribution(gradient, noise_level, clamp_range, precision, mask_threshold)
                    for gradient in gradients
                )
            else:
                attributions = gradients

        finally:
            self._remove_hooks()

        undo_gradient_requirements(inputs_tuple, gradient_mask)
        return _format_output(is_inputs_tuple, attributions)

    def _process_attribution(
        self,
        attributions: Tensor,
        noise_level: float,
        clamp_range: tuple,
        precision: int,
        mask_threshold: float,
    ) -> Tensor:
        """
        Applies privacy transformations to attributions.
        """
        # Debugging to confirm consistent parameter values

        noise = torch.randn_like(attributions) * noise_level  # ✅ Always use the explicitly passed noise_level
        attributions = attributions + noise  # Add noise
        attributions = torch.clamp(attributions, *clamp_range)  # Clamp values
        attributions = torch.round(attributions * (10**precision)) / (10**precision)  # Reduce precision
        attributions[torch.abs(attributions) < mask_threshold] = 0  # Mask small values
        return attributions

    def _register_hooks(self, module: Module):
        if isinstance(module, torch.nn.ReLU):
            hooks = _register_backward_hook(module, self._backward_hook, self)
            self.backward_hooks.extend(hooks)

    def _backward_hook(
        self,
        module: Module,
        grad_input: Union[Tensor, Tuple[Tensor, ...]],
        grad_output: Union[Tensor, Tuple[Tensor, ...]],
    ):
        to_override_grads = grad_output if self.use_relu_grad_output else grad_input
        if isinstance(to_override_grads, tuple):
            return tuple(
                F.relu(to_override_grad) for to_override_grad in to_override_grads
            )
        else:
            return F.relu(to_override_grads)

    def _remove_hooks(self):
        for hook in self.backward_hooks:
            hook.remove()


class GuidedBackprop(ModifiedReluGradientAttribution):
    """
    Computes attribution using guided backpropagation.
    """

    def __init__(self, model: Module) -> None:
        super().__init__(model, use_relu_grad_output=False)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        correc: bool = True,
        noise_level: float = 0.01,
        clamp_range: tuple = (-1, 1),
        precision: int = 4,
        mask_threshold: float = 1e-3,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Computes Guided Backpropagation attributions with privacy protections.
        """
        return super().attribute(
            inputs, target, additional_forward_args, correc, noise_level, clamp_range, precision, mask_threshold
        )


class Deconvolution(ModifiedReluGradientAttribution):
    """
    Computes attribution using deconvolution.
    """

    def __init__(self, model: Module) -> None:
        super().__init__(model, use_relu_grad_output=True)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        correc: bool = True,
        noise_level: float = 0.01,
        clamp_range: tuple = (-1, 1),
        precision: int = 4,
        mask_threshold: float = 1e-3,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Computes Deconvolution attributions with privacy protections.
        """
        return super().attribute(
            inputs, target, additional_forward_args, correc, noise_level, clamp_range, precision, mask_threshold
        )
