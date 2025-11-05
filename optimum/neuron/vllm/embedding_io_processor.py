# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Sequence
from vllm.config import VllmConfig
from vllm.entrypoints.openai.protocol import IOProcessorRequest, IOProcessorResponse
from vllm.inputs.data import PromptType
from vllm.outputs import PoolingRequestOutput, EmbeddingOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams


class NeuronEmbeddingIOProcessor(IOProcessor):
    """IO processor for Neuron embedding models."""

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)

    def parse_request(self, request: Any) -> list[str]:
        if isinstance(request, dict) and "data" in request:
            data = request["data"]
            if isinstance(data, list):
                return data
            return [data]
        if isinstance(request, str):
            return [request]
        if isinstance(request, list):
            return request
        return [str(request)]

    def pre_process(self, prompt: list[str], request_id: str | None = None, **kwargs) -> Sequence[PromptType]:
        # Each text in the batch becomes a separate vLLM request
        # Store the parent request_id to group them later
        if not hasattr(self, '_batch_map'):
            self._batch_map = {}
        if request_id:
            self._batch_map[request_id] = len(prompt)
        return [{"prompt": text} for text in prompt]

    def post_process(
        self, model_output: Sequence[PoolingRequestOutput], request_id: str | None = None, **kwargs
    ) -> list:
        from optimum.neuron.vllm.embedding_cache import get_embedding
        
        # model_output contains all outputs for this batch request
        # Return only the embeddings that belong to this request_id
        results = []
        for output in model_output:
            embedding = get_embedding(output.request_id)
            if embedding is None:
                embedding = [0.0] * 1024
            results.append(embedding)
        
        return results

    def validate_or_generate_params(
        self, params: PoolingParams | None = None
    ) -> PoolingParams:
        return params or PoolingParams()

    def output_to_response(self, plugin_output: EmbeddingOutput) -> IOProcessorResponse:
        return IOProcessorResponse(
            request_id="",
            data=plugin_output.embedding,
        )


def register():
    """Entry point for vLLM plugin system."""
    return "optimum.neuron.vllm.embedding_io_processor.NeuronEmbeddingIOProcessor"
