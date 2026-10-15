"""ONNX Engine module for generating, inspecting, and running ONNX model inference."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort


def build_linear_model(
    model_path: str,
    input_dim: int = 4,
    weights: Optional[List[float]] = None,
    bias: Optional[List[float]] = None,
    opset_version: int = 20,
    ir_version: int = 10,
) -> str:
    """Builds and saves a linear transformation ONNX model: Y = X * W + B.

    Args:
        model_path: Destination path to save .onnx file.
        input_dim: Dimension of input vector.
        weights: Weights array of length input_dim.
        bias: Bias array of length input_dim.
        opset_version: Target ONNX opset version.
        ir_version: Target ONNX IR version.

    Returns:
        Normalized string path to saved ONNX model.
    """
    if weights is None:
        weights = [2.0] * input_dim
    if bias is None:
        bias = [0.5] * input_dim

    if len(weights) != input_dim or len(bias) != input_dim:
        raise ValueError("Weights and bias must match input_dim length.")

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, input_dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, input_dim])

    w_array = np.array([weights], dtype=np.float32)
    b_array = np.array([bias], dtype=np.float32)

    W_tensor = helper.make_tensor("W", TensorProto.FLOAT, [1, input_dim], w_array.flatten())
    B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [1, input_dim], b_array.flatten())

    mul_node = helper.make_node("Mul", ["X", "W"], ["mul_out"])
    add_node = helper.make_node("Add", ["mul_out", "B"], ["Y"])

    graph = helper.make_graph(
        [mul_node, add_node],
        "linear_transform",
        [X],
        [Y],
        initializer=[W_tensor, B_tensor],
    )

    op_domain = helper.make_opsetid("", opset_version)
    model = helper.make_model(
        graph,
        producer_name="sovereign-onnx-engine",
        ir_version=ir_version,
        opset_imports=[op_domain],
    )

    norm_path = os.path.normpath(model_path)
    os.makedirs(os.path.dirname(norm_path) if os.path.dirname(norm_path) else ".", exist_ok=True)
    onnx.save(model, norm_path)
    return norm_path


def inspect_onnx_model(model_path: str) -> Dict[str, Any]:
    """Inspects an ONNX model file and returns its graph metadata."""
    norm_path = os.path.normpath(model_path)
    model = onnx.load(norm_path)

    inputs = [node.name for node in model.graph.input]
    outputs = [node.name for node in model.graph.output]
    nodes = [node.op_type for node in model.graph.node]
    initializers = [init.name for init in model.graph.initializer]

    return {
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
        "initializers": initializers,
    }


def run_onnx_inference(
    model_path: str,
    input_data: Union[List[float], np.ndarray],
    input_name: str = "X",
) -> np.ndarray:
    """Runs inference on the specified ONNX model using ONNX Runtime.

    Args:
        model_path: Path to ONNX model.
        input_data: List or NumPy array of float values.
        input_name: Name of input tensor node.

    Returns:
        NumPy array of model inference outputs.
    """
    norm_path = os.path.normpath(model_path)
    session = ort.InferenceSession(norm_path, providers=["CPUExecutionProvider"])

    if isinstance(input_data, list):
        data_arr = np.array([input_data], dtype=np.float32)
    else:
        data_arr = np.asarray(input_data, dtype=np.float32)
        if data_arr.ndim == 1:
            data_arr = np.expand_dims(data_arr, axis=0)

    outputs = session.run(None, {input_name: data_arr})
    return outputs[0]
