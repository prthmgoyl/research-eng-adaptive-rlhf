# JITRL Quick Start Guide

Get started with Just-In-Time Reinforcement Learning in vLLM in 5 minutes.

## Installation

```bash
# vLLM with all required dependencies is already installed
# Just make sure you have the latest version
pip install -U vllm
```

## 1. Start vLLM with JITRL (Simple Setup)

The easiest way to test JITRL without a trainer:

```bash
# Enable JITRL feedback collection (no trainer)
vllm serve Qwen/Qwen3-1.7B \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-jitrl-feedback \
    --jitrl-feedback-batch-size 10
```

This will:
- Enable the `/v1/feedback` endpoint
- Collect feedback in a buffer (size: 10)
- Log when buffer is full (but won't trigger updates without trainer)

## 2. Test the Feedback API

```bash
# Terminal 1: Start vLLM (from step 1)

# Terminal 2: Generate a completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "Explain photosynthesis"}]
  }' | jq '.id'

# Save the request ID, then submit feedback
curl -X POST http://localhost:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "chatcmpl-abc123",
    "reward": 0.9
  }' | jq '.'

# Check buffer status
curl http://localhost:8000/v1/feedback/status | jq '.'
```

## 3. Add a Trainer (Full Setup)

To enable automatic weight updates, add a trainer:

### 3a. Start vLLM with Trainer URL

```bash
export VLLM_SERVER_DEV_MODE=1  # Required for pause/resume

vllm serve Qwen/Qwen3-1.7B \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-jitrl-feedback \
    --jitrl-feedback-batch-size 5 \
    --jitrl-trainer-url http://localhost:8300 \
    --weight-transfer-backend nccl \
    --gpu-memory-utilization 0.7
```

### 3b. Start the Trainer

```bash
# In a new terminal
cd examples/jitrl

python trainer_jitrl.py \
    --model Qwen/Qwen3-1.7B \
    --host 0.0.0.0 \
    --port 8300 \
    --gpu-id 1  # Use a different GPU than vLLM
```

### 3c. Run the Example Client

```bash
# In another terminal
cd examples/jitrl

python client_example.py \
    --base-url http://localhost:8000 \
    --num-requests 10
```

You should see:
1. Requests being generated
2. Feedback being submitted
3. Weight update triggered after 5 feedback entries
4. vLLM pausing, trainer running, weights syncing
5. vLLM resuming with updated weights

## 4. Use from Python

```python
import openai
import requests

# Initialize client
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

# Generate response
response = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
)

print(f"Response: {response.choices[0].message.content}")
print(f"Request ID: {response.id}")

# Submit positive feedback
feedback_response = requests.post(
    "http://localhost:8000/v1/feedback",
    json={
        "request_id": response.id,
        "reward": 0.95,  # High reward!
        "feedback_text": "Excellent explanation",
    },
)

print(f"Feedback status: {feedback_response.json()}")
```

## 5. Monitor Buffer Status

### Via API
```bash
curl http://localhost:8000/v1/feedback/status | jq '.'
```

### Via Python
```python
import requests

status = requests.get("http://localhost:8000/v1/feedback/status").json()
print(f"Buffer: {status['buffer_size']}/{status['buffer_capacity']}")
```

## 6. Manual Weight Updates

Trigger an update without waiting for buffer to fill:

```bash
curl -X POST http://localhost:8000/v1/feedback/trigger_update | jq '.'
```

## Common Configuration

### Small Batch for Testing
```bash
--jitrl-feedback-batch-size 3  # Update after just 3 feedback entries
```

### Production Settings
```bash
--jitrl-feedback-batch-size 128  # Larger batches for efficiency
--jitrl-auto-update true         # Auto-trigger updates
```

### Disable Auto-Updates
```bash
--jitrl-auto-update false  # Manual control only
```

Then trigger manually:
```bash
curl -X POST http://localhost:8000/v1/feedback/trigger_update
```

## Architecture Options

### Option 1: Single Node (Simplest)
```
┌────────────────────────────┐
│  Same Machine              │
│  ├─ GPU 0: vLLM Server     │
│  └─ GPU 1: Trainer         │
└────────────────────────────┘
```

### Option 2: Separate Nodes
```
┌──────────────┐    ┌──────────────┐
│ Node 1       │    │ Node 2       │
│ vLLM Server  │───►│ Trainer      │
│ (Multi-GPU)  │    │ (GPU)        │
└──────────────┘    └──────────────┘
```

### Option 3: Disaggregated
```
┌────────┐  ┌────────┐  ┌─────────┐
│Prefill │  │ Decode │  │ Trainer │
│ Node   │  │  Node  │  │  Node   │
└────────┘  └────────┘  └─────────┘
     └─────────┴────────────┘
            Proxy
```

## Troubleshooting

### "JITRL feedback is not enabled"
- Make sure you started vLLM with `--enable-jitrl-feedback`

### Feedback submitted but no updates
- Check if trainer URL is set: `--jitrl-trainer-url http://...`
- Verify trainer is running: `curl http://localhost:8300/health`
- Check buffer size: May not be full yet

### "Connection refused" to trainer
- Ensure trainer is running and accessible
- Check firewall rules if using multiple nodes
- Verify port is not already in use: `lsof -i :8300`

### NCCL errors
- Set `export NCCL_DEBUG=INFO` for verbose logs
- Ensure GPUs are visible: `nvidia-smi`
- Check network connectivity between nodes

### Weight updates taking too long
- Reduce feedback batch size for faster updates
- Use `ipc` backend instead of `nccl` for same-GPU
- Optimize training loop in trainer

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture details
- Customize [trainer_jitrl.py](trainer_jitrl.py) with your RL algorithm
- Integrate with your application's feedback collection system

## Example Applications

### Chatbot with User Ratings
```python
def handle_user_message(message):
    # Generate response
    response = generate_response(message)
    
    # Show to user with rating buttons
    display_response(response)
    
    # Collect rating
    rating = wait_for_user_rating()  # 1-5 stars
    
    # Convert to reward (0.0-1.0)
    reward = (rating - 1) / 4.0
    
    # Submit feedback
    submit_feedback(response.id, reward)
```

### A/B Testing
```python
def ab_test_response(prompt):
    # Generate two responses
    response_a = model_a.generate(prompt)
    response_b = model_b.generate(prompt)
    
    # Show both to user
    choice = user_chooses_best(response_a, response_b)
    
    # Submit feedback
    submit_feedback(response_a.id, reward=1.0 if choice == 'a' else 0.0)
    submit_feedback(response_b.id, reward=1.0 if choice == 'b' else 0.0)
```

### Implicit Feedback
```python
def track_engagement(response):
    # Monitor user behavior
    time_spent = measure_reading_time()
    clicked_links = count_link_clicks()
    copied_text = detect_copy_actions()
    
    # Compute implicit reward
    reward = compute_engagement_score(
        time_spent, clicked_links, copied_text
    )
    
    # Submit feedback
    submit_feedback(response.id, reward)
```

## Support

- GitHub Issues: Tag with `jitrl` or `rlhf`
- Documentation: [vLLM RLHF Docs](../../docs/training/rlhf.md)
- Discord: #rlhf channel
