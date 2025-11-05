import time
import random
from vllm import LLM
from vllm.engine.llm_engine import LLMEngine
from vllm.pooling_params import PoolingParams as VllmPoolingParams
from vllm.sampling_params import SamplingParams

# Monkey-patch the engine to accept PoolingParams
_original_add_processed_request = LLMEngine._add_processed_request

def _patched_add_processed_request(self, request_id, processed_inputs, params, arrival_time, lora_request, trace_headers=None, priority=0):
    if isinstance(params, VllmPoolingParams):
        params = SamplingParams(temperature=0, max_tokens=1)
    return _original_add_processed_request(self, request_id, processed_inputs, params, arrival_time, lora_request, trace_headers, priority)

LLMEngine._add_processed_request = _patched_add_processed_request

llm = LLM(
    model="qwen3-embedding-0.6b-neuron",
    max_num_seqs=4,
    max_model_len=1024,
    tensor_parallel_size=4,
    runner="pooling",
    io_processor_plugin="embedding",
)

# Test data
test_texts = [
    "AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"
]

num_requests = 500
batch_size = 4

print(f"Starting stress test: {num_requests} requests with batch size {batch_size}")
start_time = time.time()

successful = 0
failed = 0

for i in range(num_requests):
    # Random batch of texts
    batch = random.sample(test_texts, min(batch_size, len(test_texts)))
    
    try:
        prompts = {"data": batch}
        outputs = llm.encode(prompts)
        
        # Verify outputs
        assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
        # outputs[0].outputs is a list where each element is an embedding vector
        embeddings = outputs[0].outputs
        assert len(embeddings) == len(batch), f"Expected {len(batch)} embeddings, got {len(embeddings)}"
        
        for emb in embeddings:
            assert isinstance(emb, list), f"Expected list, got {type(emb)}"
            assert len(emb) > 0, "Empty embedding"
        
        successful += 1
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_requests} requests")
    except Exception as e:
        failed += 1
        print(f"Request {i} failed: {e}")

end_time = time.time()
elapsed = end_time - start_time

print(f"\n{'='*60}")
print(f"Stress Test Results:")
print(f"{'='*60}")
print(f"Total requests: {num_requests}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Time elapsed: {elapsed:.2f}s")
print(f"Requests/sec: {num_requests/elapsed:.2f}")
print(f"{'='*60}")
