# Just-In-Time Reinforcement Learning (JITRL) with vLLM

This directory contains examples for using vLLM's JITRL (Just-In-Time Reinforcement Learning) support, which enables online reinforcement learning with real-time feedback and weight updates.

## Overview

JITRL allows you to:
1. **Collect user feedback** on model responses in real-time via the `/v1/feedback` endpoint
2. **Accumulate rewards** in a feedback buffer
3. **Trigger automatic weight updates** when the buffer reaches capacity
4. **Coordinate updates** across distributed inference nodes (prefill/decode disaggregation)

## Architecture

```
┌──────────┐      ┌──────────────┐      ┌────────────┐
│  Client  │─────►│  vLLM Server │─────►│  Training  │
│          │      │              │      │   Node     │
└──────────┘      └──────────────┘      └────────────┘
     │                   │                      │
     │  1. Generate      │                      │
     │  Response         │                      │
     │◄──────────────────│                      │
     │                   │                      │
     │  2. Submit        │                      │
     │  Feedback         │                      │
     │──────────────────►│                      │
     │                   │                      │
     │                   │  3. Send Feedback    │
     │                   │  Batch (when full)   │
     │                   │─────────────────────►│
     │                   │                      │
     │                   │  4. Broadcast        │
     │                   │  Updated Weights     │
     │                   │◄─────────────────────│
     │                   │  (via NCCL)          │
     │                   │                      │
```

## Quick Start

### 1. Start vLLM with JITRL Support

```bash
# Enable dev mode for pause/resume endpoints
export VLLM_SERVER_DEV_MODE=1

# Start vLLM server with JITRL feedback enabled
vllm serve Qwen/Qwen3-1.7B \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-jitrl-feedback \
    --jitrl-feedback-batch-size 32 \
    --jitrl-trainer-url http://localhost:8300 \
    --weight-transfer-backend nccl
```

### 2. Start the JITRL Trainer

The trainer receives feedback batches, performs training updates, and broadcasts weights:

```bash
python trainer_jitrl.py \
    --model Qwen/Qwen3-1.7B \
    --host 0.0.0.0 \
    --port 8300 \
    --gpu-id 1
```

### 3. Use the API

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

# Generate a response
response = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
)

request_id = response.id
response_text = response.choices[0].message.content

# Submit feedback with reward
import requests

feedback = {
    "request_id": request_id,
    "reward": 0.9,  # High reward for good response
}

r = requests.post(
    "http://localhost:8000/v1/feedback",
    json=feedback,
)
print(f"Feedback status: {r.json()}")
```

## API Reference

### POST `/v1/feedback`

Submit feedback for a chat completion response.

**Request Body:**
```json
{
    "request_id": "chatcmpl-abc123",      // Required: response ID from /v1/chat/completions
    "conversation_id": "conv-456",        // Optional: for multi-turn tracking
    "turn_index": 0,                      // Optional: turn number in conversation
    "reward": 0.8,                        // Required: reward score (typically 0.0-1.0)
    "feedback_text": "Great response!"    // Optional: human-readable feedback
}
```

**Response:**
```json
{
    "status": "accepted",
    "buffer_size": 15,
    "buffer_capacity": 32,
    "update_triggered": false
}
```

### GET `/v1/feedback/status`

Get current feedback buffer status.

**Response:**
```json
{
    "buffer_size": 15,
    "buffer_capacity": 32,
    "trainer_url": "http://localhost:8300",
    "auto_update": true
}
```

### POST `/v1/feedback/trigger_update`

Manually trigger a weight update regardless of buffer size.

**Response:**
```json
{
    "status": "update_triggered",
    "feedback_count": 15
}
```

## Configuration Options

### vLLM Server Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-jitrl-feedback` | Enable JITRL feedback collection | `False` |
| `--jitrl-feedback-batch-size` | Feedback entries to accumulate before update | `32` |
| `--jitrl-trainer-url` | URL of the JITRL trainer server | `None` |
| `--jitrl-auto-update` | Auto-trigger updates when buffer is full | `True` |
| `--weight-transfer-backend` | Weight sync backend (`nccl` or `ipc`) | - |

### Trainer Configuration

The trainer script accepts:
- `--model`: Model name or path
- `--host`: Trainer server host
- `--port`: Trainer server port  
- `--gpu-id`: GPU device ID for training

## Advanced Usage

### Disaggregated Prefill/Decode with JITRL

For disaggregated serving, use the specialized proxy:

```bash
# Start prefill node
vllm serve MODEL --port 8100 --weight-transfer-backend nccl

# Start decode node
vllm serve MODEL --port 8200 --weight-transfer-backend nccl

# Start trainer
python trainer_jitrl.py --model MODEL --port 8300

# Start proxy with JITRL
python disagg_proxy_jitrl.py \
    --prefiller-host localhost --prefiller-port 8100 \
    --decoder-host localhost --decoder-port 8200 \
    --enable-jitrl \
    --trainer-host localhost --trainer-port 8300 \
    --feedback-batch-size 32
```

### Custom Training Logic

The `trainer_jitrl.py` includes a placeholder training function. Replace it with your actual RL algorithm:

```python
async def train_on_feedback(self, feedback_batch: list[dict[str, Any]]) -> None:
    """Implement your RL algorithm here:
    - PPO (Proximal Policy Optimization)
    - DPO (Direct Preference Optimization)  
    - GRPO (Group Relative Policy Optimization)
    - RLHF with reward model
    - etc.
    """
    # Your training code here
    pass
```

### Multi-Turn Conversations

Track conversation state across turns using `conversation_id`:

```python
conversation_id = "conv-123"

# Turn 1
response1 = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[{"role": "user", "content": "Hello"}],
)

requests.post("http://localhost:8000/v1/feedback", json={
    "request_id": response1.id,
    "conversation_id": conversation_id,
    "turn_index": 0,
    "reward": 0.9,
})

# Turn 2  
response2 = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": response1.choices[0].message.content},
        {"role": "user", "content": "Tell me more"},
    ],
)

requests.post("http://localhost:8000/v1/feedback", json={
    "request_id": response2.id,
    "conversation_id": conversation_id,
    "turn_index": 1,
    "reward": 0.85,
})
```

## Examples

- `client_example.py` - Basic JITRL client with feedback submission
- `disagg_proxy_jitrl.py` - Disaggregated serving proxy with JITRL
- `trainer_jitrl.py` - Example JITRL trainer server

## Troubleshooting

### Weight Updates Not Triggering

1. Check that `VLLM_SERVER_DEV_MODE=1` is set
2. Verify trainer URL is accessible from vLLM server
3. Check feedback buffer size: `curl http://localhost:8000/v1/feedback/status`
4. Review vLLM logs for connection errors

### NCCL Errors

1. Ensure all nodes can reach each other
2. Verify GPU visibility with `nvidia-smi`
3. Check that world_size matches (1 trainer + N vLLM instances)
4. Set `NCCL_DEBUG=INFO` for detailed NCCL logs

### Slow Weight Updates

1. Reduce feedback batch size for more frequent updates
2. Use `ipc` backend instead of `nccl` for single-GPU setups
3. Optimize training loop in `trainer_jitrl.py`
4. Monitor trainer logs for bottlenecks

## Related Documentation

- [vLLM Weight Transfer](../../docs/training/weight_transfer/README.md)
- [Async Reinforcement Learning](../../docs/training/async_rl.md)
- [RLHF with vLLM](../../docs/training/rlhf.md)
