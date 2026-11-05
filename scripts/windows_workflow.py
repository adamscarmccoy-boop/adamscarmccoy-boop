"""Integrated Windows High-Performance Acceleration Workflow.

Combines Native C dynamic library math processing, Rust + Apache Arrow shared objects,
ONNX Runtime tensor evaluation, and Local Llama / LM Studio LLM text generation
into a unified high-velocity workflow pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa

from scripts.dll_runner import NativeBridgeRunner
from scripts.llama_runner import UnifiedLlamaRunner
from scripts.onnx_engine import build_linear_model, run_onnx_inference
from scripts.rust_arrow_bridge import RustArrowBridgeRunner


class WindowsWorkflowOrchestrator:
    """Orchestrates native C execution, Rust + Arrow shared object processing, ONNX inference, and LLM text generation."""

    def __init__(
        self,
        onnx_model_path: str = "build/pipeline_model.onnx",
        native_lib_path: Optional[str] = None,
        rust_lib_path: Optional[str] = None,
        llm_mode: str = "lm_studio",
        llm_base_url: str = "http://127.0.0.1:1234/v1",
        llm_model: str = "default",
        llama_exe_path: Optional[str] = None,
        gguf_model_path: Optional[str] = None,
        use_rust_arrow: bool = True,
    ):
        self.onnx_model_path = os.path.normpath(onnx_model_path)
        self.native_runner = NativeBridgeRunner(lib_path=native_lib_path)
        self.use_rust_arrow = use_rust_arrow
        self.rust_arrow_runner = (
            RustArrowBridgeRunner(lib_path=rust_lib_path) if use_rust_arrow else None
        )
        self.llm_runner = UnifiedLlamaRunner(
            mode=llm_mode,
            base_url=llm_base_url,
            model_name=llm_model,
            llama_exe_path=llama_exe_path,
            gguf_model_path=gguf_model_path,
        )

    def prepare_onnx_pipeline(self, weights: List[float], bias: List[float]) -> str:
        """Builds and initializes the ONNX model for high-speed inference."""
        dim = len(weights)
        return build_linear_model(
            model_path=self.onnx_model_path,
            input_dim=dim,
            weights=weights,
            bias=bias,
        )

    def execute_pipeline(
        self,
        raw_input: Union[List[float], pa.Array, Any],
        weights: List[float],
        bias: List[float],
        llm_prompt_template: str = "Analyze these calculation metrics and loss score: {metrics}",
    ) -> Dict[str, Any]:
        """Runs full integrated workflow across C native DLL, Rust + Arrow shared object, ONNX model, and LLM runner."""
        # Convert input to PyArrow array if necessary
        if isinstance(raw_input, pa.Array):
            arrow_input = raw_input
            input_list = raw_input.to_pylist()
        else:
            input_list = [float(x) for x in raw_input]
            arrow_input = pa.array(input_list, type=pa.float64())

        # 1. Acceleration via C native DLL or Rust + Apache Arrow shared object
        if self.use_rust_arrow and self.rust_arrow_runner is not None:
            scaled_input = self.rust_arrow_runner.vector_scale(arrow_input, scale=1.5)
            dot_score = self.rust_arrow_runner.dot_product(arrow_input, scaled_input)
        else:
            scaled_input = self.native_runner.vector_scale(input_list, scale=1.5)
            dot_score = self.native_runner.dot_product(input_list, scaled_input)

        # 2. ONNX model inference
        if not os.path.exists(self.onnx_model_path):
            self.prepare_onnx_pipeline(weights, bias)

        onnx_output = run_onnx_inference(self.onnx_model_path, scaled_input)
        onnx_flat = onnx_output.flatten().tolist()

        # 3. Loss computation between input and ONNX prediction
        if self.use_rust_arrow and self.rust_arrow_runner is not None:
            loss_val = self.rust_arrow_runner.compute_loss(arrow_input, onnx_flat)
        else:
            loss_val = self.native_runner.compute_loss(input_list, onnx_flat)

        metrics_summary = (
            f"Input: {input_list}, Scaled: {scaled_input}, Dot Product: {dot_score:.4f}, "
            f"ONNX Output: {onnx_flat}, Loss: {loss_val:.6f}"
        )

        prompt = llm_prompt_template.format(metrics=metrics_summary)

        # 4. LLM Generation
        llm_response = ""
        try:
            llm_response = self.llm_runner.generate(prompt)
        except Exception as err:
            llm_response = f"[LLM Offline / Fallback] Workflow completed metrics: {metrics_summary}. Error: {err}"

        return {
            "scaled_input": scaled_input,
            "dot_score": dot_score,
            "onnx_output": onnx_flat,
            "loss": loss_val,
            "metrics_summary": metrics_summary,
            "llm_response": llm_response,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated Windows workflow.")
    parser.add_argument("--onnx-model", default="build/pipeline_model.onnx")
    parser.add_argument("--llm-mode", default="lm_studio", choices=["lm_studio", "llama_server", "llama_exe"])
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="default")
    parser.add_argument("--no-rust", action="store_true", help="Disable Rust + Arrow engine")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orchestrator = WindowsWorkflowOrchestrator(
        onnx_model_path=args.onnx_model,
        llm_mode=args.llm_mode,
        llm_base_url=args.base_url,
        llm_model=args.model,
        use_rust_arrow=not args.no_rust,
    )
    res = orchestrator.execute_pipeline(
        raw_input=[1.0, 2.0, 3.0, 4.0],
        weights=[2.0, 2.0, 2.0, 2.0],
        bias=[0.5, 0.5, 0.5, 0.5],
    )
    print("=== Workflow Result ===")
    print(f"Metrics Summary: {res['metrics_summary']}")
    print(f"LLM Response:\n{res['llm_response']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
