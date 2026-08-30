# Just-In-Time Reinforcement Learning (JITRL)

vLLM supports Just-In-Time Reinforcement Learning (JITRL), which enables online reinforcement learning with real-time feedback collection and weight updates during inference.

## Overview

JITRL allows you to:

- **Collect user feedback** on model responses in real-time via the `/v1/feedback` endpoint
- **Accumulate rewards** in a feedback buffer
- **Trigger automatic weight updates** when the buffer reaches capacity
- **Coordinate updates** across distributed inference nodes

This enables continuous model improvement based on user interactions without interrupting the serving process.

## Quick Start

### Enable JITRL Feedback

Start vLLM with JITRL feedback enabled:

```bash
export VLLM_SERVER_DEV_MODE=1  # Required for pause/resume

vllm serve MODEL_NAME \
    --enable-jitrl-feedback \
    --jitrl-feedback-batch-size 32 \
    --jitrl-trainer-url http://localhost:8300 \
    --weight-transfer-backend nccl
```

### Submit Feedback

After generating a response, submit feedback with a reward:

```python
import openai
import requests

# Generate response
client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="MODEL_NAME",
    messages=[{"role": "user", "content": "Your prompt"}],
)

# Submit feedback
requests.post(
    "http://localhost:8000/v1/feedback",
    json={
        "request_id": response.id,
        "reward": 0.9,  # 0.0-1.0 or your reward range
    },
)
```

## API Reference

### POST `/v1/feedback`

Submit feedback for a chat completion response.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string | Yes | The ID from the chat completion response |
| `reward` | float | Yes | Reward score (typically 0.0-1.0) |
| `conversation_id` | string | No | For multi-turn conversation tracking |
| `turn_index` | integer | No | Turn number within the conversation |
| `feedback_text` | string | No | Human-readable feedback |

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

Get the current feedback buffer status.

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

## Configuration

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-jitrl-feedback` | flag | False | Enable JITRL feedback collection |
| `--jitrl-feedback-batch-size` | int | 32 | Buffer size before triggering updates |
| `--jitrl-trainer-url` | string | None | URL of the JITRL trainer server |
| `--jitrl-auto-update` | flag | True | Auto-trigger updates when buffer is full |
| `--weight-transfer-backend` | string | - | Backend for weight sync (`nccl` or `ipc`) |

## Architecture

### Weight Update Flow

```
1. Client generates responses
   └─> vLLM caches response by request_id

2. Client submits feedback
   └─> Feedback added to buffer
       └─> When buffer full:
           ├─> Pause vLLM generation (mode="keep")
           ├─> Send feedback to trainer
           ├─> Trainer performs RL update
           ├─> Trainer broadcasts weights (NCCL)
           └─> Resume vLLM generation
```

### Trainer Server

The trainer server:
- Receives feedback batches via HTTP `/train` endpoint
- Performs RL training (PPO, DPO, GRPO, etc.)
- Broadcasts updated weights to vLLM via NCCL
- Provides sync status via `/sync_status` endpoint

Example trainer implementation: `examples/jitrl/trainer_jitrl.py`

## Use Cases

### Explicit User Feedback

```python
def handle_user_rating(response_id, star_rating):
    """Convert user's star rating to reward."""
    reward = (star_rating - 1) / 4.0  # 1-5 stars -> 0.0-1.0
    
    requests.post(
        f"{base_url}/v1/feedback",
        json={"request_id": response_id, "reward": reward},
    )
```

### Implicit Feedback

```python
def track_engagement(response_id):
    """Use engagement metrics as implicit reward."""
    time_spent = measure_reading_time()
    clicked_links = count_link_clicks()
    
    # Compute reward from engagement
    reward = compute_engagement_score(time_spent, clicked_links)
    
    requests.post(
        f"{base_url}/v1/feedback",
        json={"request_id": response_id, "reward": reward},
    )
```

### Multi-Turn Conversations

```python
conversation_id = "conv-123"

# Track feedback across turns
for turn_idx, user_message in enumerate(conversation):
    response = client.chat.completions.create(
        model=MODEL,
        messages=conversation,
    )
    
    # Get user feedback for this turn
    reward = get_user_feedback()
    
    requests.post(
        f"{base_url}/v1/feedback",
        json={
            "request_id": response.id,
            "conversation_id": conversation_id,
            "turn_index": turn_idx,
            "reward": reward,
        },
    )
```

## Advanced Usage

### Disaggregated Serving

For disaggregated prefill/decode setups, use the specialized proxy:

```bash
# Start prefill and decode nodes with weight transfer
vllm serve MODEL --port 8100 --weight-transfer-backend nccl
vllm serve MODEL --port 8200 --weight-transfer-backend nccl

# Start proxy with JITRL
python examples/disaggregated/disaggregated_serving/disagg_proxy_jitrl.py \
    --prefiller-host localhost --prefiller-port 8100 \
    --decoder-host localhost --decoder-port 8200 \
    --enable-jitrl \
    --trainer-host localhost --trainer-port 8300
```

### Custom Training Algorithms

Implement your RL algorithm in the trainer's `train_on_feedback()` method:

```python
async def train_on_feedback(self, feedback_batch):
    """
    Replace with your RL algorithm:
    - PPO (Proximal Policy Optimization)
    - DPO (Direct Preference Optimization)
    - GRPO (Group Relative Policy Optimization)
    - RLHF with reward model
    """
    # Your training code here
    pass
```

## Best Practices

### Reward Design

1. **Normalize rewards** to a consistent range (e.g., 0.0-1.0)
2. **Balance positive and negative** feedback
3. **Consider delayed rewards** for long-term outcomes
4. **Use multiple reward sources** (explicit + implicit)

### Batch Size Selection

- **Small batches (5-10)**: Faster adaptation, more frequent updates
- **Medium batches (32-64)**: Balanced efficiency and stability
- **Large batches (128+)**: Better training stability, less overhead

### Production Considerations

1. **Monitor buffer status** regularly
2. **Log feedback submissions** for analysis
3. **Track reward distributions** to detect issues
4. **Set up alerts** for trainer failures
5. **Use redundant trainers** for high availability

## Troubleshooting

### Feedback Not Triggering Updates

- Verify `--enable-jitrl-feedback` is set
- Check trainer URL is accessible
- Ensure `VLLM_SERVER_DEV_MODE=1` is set
- Review logs for connection errors

### Weight Update Failures

- Check NCCL configuration and GPU visibility
- Verify world_size matches (1 trainer + N vLLM workers)
- Ensure sufficient GPU memory for both serving and training
- Set `NCCL_DEBUG=INFO` for detailed logs

### Slow Updates

- Reduce feedback batch size
- Use `ipc` backend for single-GPU setups
- Optimize training loop
- Profile trainer with PyTorch profiler

## Related Documentation

- [Weight Transfer](../training/weight_transfer/README.md) - NCCL/IPC weight sync
- [Async Reinforcement Learning](../training/async_rl.md) - Pause/resume API
- [RLHF](../training/rlhf.md) - RL libraries using vLLM

## Examples

Complete examples are available in `examples/jitrl/`:

- `client_example.py` - Basic JITRL client
- `trainer_jitrl.py` - Example trainer server
- `disagg_proxy_jitrl.py` - Disaggregated serving with JITRL
- `README.md` - Comprehensive guide
- `QUICKSTART.md` - 5-minute quick start

## Limitations

- Requires `VLLM_SERVER_DEV_MODE=1` for pause/resume endpoints
- Trainer must be reachable from vLLM server
- Weight updates pause inference (brief interruption)
- NCCL requires GPU-to-GPU connectivity
- Response caching limited to 1000 most recent requests

## References

- [TRL GRPO with vLLM](https://huggingface.co/learn/cookbook/grpo_vllm_online_training)
- [Unsloth + vLLM GRPO](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - RL library with vLLM support
