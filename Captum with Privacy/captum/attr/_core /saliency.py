#!/usr/bin/env python3

# pyre-strict

from typing import Callable, Optional
import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.gradient import apply_gradient_requirements, undo_gradient_requirements
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage
from torch import Tensor


class Saliency(GradientAttribution):
    """
    A baseline approach for computing input attribution. It returns
    the gradients with respect to inputs. Now includes minor privacy safeguards.

    More details about the approach can be found in the following paper:
        https://arxiv.org/abs/1312.6034
    """

    def __init__(self, forward_func: Callable[..., Tensor]) -> None:
        """
        Args:
            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        super().__init__(forward_func)
        # Store instance variables for consistency
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
        abs: bool = True,
        additional_forward_args: Optional[object] = None,
        correc: bool = True,
        noise_level: float = 0.01,  # Noise added to attributions
        clamp_range: tuple = (-1, 1),  # Limits extreme values
        precision: int = 4,  # Rounds to limit precision
        mask_threshold: float = 1e-3,  # Zeroes out small attributions
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Computes saliency with privacy protection.
        """
        # Store parameters as instance variables to maintain consistency
        self.correc = correc
        self.noise_level = noise_level
        self.clamp_range = clamp_range
        self.precision = precision
        self.mask_threshold = mask_threshold

        is_inputs_tuple = _is_tuple(inputs)
        inputs_tuple = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs_tuple)

        gradients = self.gradient_func(
            self.forward_func, inputs_tuple, target, additional_forward_args
        )

        # Debugging: Confirm correct values

        if self.correc:
            attributions = tuple(
                self._process_attribution(
                    torch.abs(gradient) if abs else gradient
                )
                for gradient in gradients
            )
        elif abs:
            attributions = tuple(torch.abs(gradient) for gradient in gradients)
        else:
            attributions = gradients

        undo_gradient_requirements(inputs_tuple, gradient_mask)
        return _format_output(is_inputs_tuple, attributions)


    def _process_attribution(self, attributions: Tensor) -> Tensor:
        """
        Applies privacy transformations to attributions.
        """


        noise = torch.randn_like(attributions) * self.noise_level  # ✅ Always use self.noise_level
        attributions = attributions + noise  # Add noise
        attributions = torch.clamp(attributions, *self.clamp_range)  # Clamp values
        attributions = torch.round(attributions * (10**self.precision)) / (10**self.precision)  # Reduce precision
        attributions[torch.abs(attributions) < self.mask_threshold] = 0  # Mask small values
        return attributions

    def attribute_future(self) -> Callable:
        """
        This method is not implemented for Saliency.
        """
        raise NotImplementedError("attribute_future is not implemented for Saliency")
