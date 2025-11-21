#!/usr/bin/env python3

"""Load a ACE model checkpoint and trace it for interoperability with FTORCH"""

import os
from pathlib import Path
from typing import Optional

import torch

from fme.ace.stepper import load_stepper


def trace_to_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filename: Optional[str] = "traced_model.pt",
) -> None:
    """
    Save PyTorch model to TorchScript using tracing.

    Parameters
    ----------
    model : torch.NN.Module
        a PyTorch model
    dummy_input : torch.Tensor
        appropriate size Tensor to act as input to model
    filename : str
        name of file to save to
    """
    # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
    traced_model = torch.jit.trace(model, dummy_input)
    frozen_model = torch.jit.freeze(traced_model)
    frozen_model.save(filename)


def load_torchscript(filename: Optional[str] = "saved_model.pt") -> torch.nn.Module:
    """
    Load a TorchScript from file.

    Parameters
    ----------
    filename : str
        name of file containing TorchScript model
    """
    model = torch.jit.load(filename)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device_type",
        help="Device type to run the inference on",
        type=str,
        choices=["cpu", "cuda", "hip", "xpu", "mps"],
        default="cpu",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to the file containing the PyTorch model",
        type=Path,
    )
    parsed_args = parser.parse_args()
    device_type = parsed_args.device_type
    checkpoint = parsed_args.checkpoint
    traced_torch_script = Path(str(checkpoint).replace(".tar", "_traced.tar"))

    # validate arguments
    if not checkpoint.exists():
        msg = f"Error: checkpoint file '{checkpoint}' does not exist."
        raise FileNotFoundError(msg)
    if not checkpoint.is_file():
        msg = f"Error: Path '{checkpoint}' is not a checkpoint file."
        raise IsADirectoryError(msg)

    # Load checkpoint and prepare for tracing
    stepper = load_stepper(checkpoint)

    # Prepare dummy input
    img_shape = stepper.get_state()['dataset_info']['img_shape']
    input_names = stepper.config.input_names

    example_input = torch.rand(1, len(input_names), *img_shape)

    # Transfer the model and inputs to requested device
    if device_type == "hip":
        device = torch.device("cuda")  # NOTE: HIP is treated as CUDA in FTorch
    else:
        device = torch.device(device_type)

    if len(stepper.modules) != 1:
        msg = ("")
        raise RuntimeError(msg)

    trained_model = stepper.modules[0].to(device)
    trained_model.eval()
    example_input = example_input.to(device)

    # Run model for dummy inputs, should catch any high-level issues
    example_outputs = trained_model(example_input)

    # Save model
    trace_to_torchscript(
        trained_model, example_input, filename=traced_torch_script
    )

    # =====================================================
    # Check model saved OK
    # =====================================================

    # Load torchscript and run model as a test
    example_input = 2.0 * example_input
    trained_model_testing_output = trained_model(example_input)

    ts_model = load_torchscript(filename=traced_torch_script)
    ts_model_output = ts_model(example_input)

    if torch.all(ts_model_output.eq(trained_model_testing_output)):
        print("Saved TorchScript model working as expected in a basic test.")
        print("Users should perform further validation as appropriate.")
    else:
        model_error = (
            "Saved Torchscript model is not performing as expected.\n"
            "Consider using scripting if you used tracing, or investigate further."
        )
        raise RuntimeError(model_error)
