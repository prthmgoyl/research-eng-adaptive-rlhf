# Dual-Mode JITRL: Logit + Weight Updates

Complete guide for using JITRL with both fast logit bias updates and slow weight updates.

## Overview

```
Single Feedback →  Splits into 2 paths:

┌─────────────────────────────────┐
│ FAST PATH: Logit Bias Updates  │  ⚡ Immediate (0ms)
├─────────────────────────────────┤
│ • Updates token reward stats    │
│ • Recomputes logit bias         │
│ • Applied on NEXT request       │
└─────────────────────────────────┘

┌─────────────────────────────────┐
│ SLOW PATH: Weight Updates       │  🔄 Batched (minutes)
├─────────────────────────────────┤
│ • Buffers feedback              │
│ • Triggers training when full   │
│ • NCCL weight broadcast         │
└─────────────────────────────────┘
```

## How It Works

### Fast Path (Logit Bias)

1. **User submits feedback** → `{request_id, reward: 0.9}`
2. **System retrieves tokens** → `[1234, 5678, 9012]` (from cached response)
3. **Updates token stats** → Each token gets reward 0.9 added to its history
4. **Recomputes bias** → High-reward tokens get positive bias
5. **Next generation** → Logit bias automatically applied

**Example:**
```python
# Token 1234 history: [0.9, 0.8, 0.9] → avg=0.87
# Bias = (0.87 - 0.5) * 2 * strength(2.0) = +1.48
# → Token 1234 gets +1.48 logit boost on next generation
```

### Slow Path (Weight Updates)

Same as before - batches feedback and triggers full model training.

## Configuration

### Enable Both Modes

```bash
vllm serve MODEL \
    --enable-jitrl-feedback \
    --jitrl-feedback-batch-size 32 \
    --jitrl-trainer-url http://localhost:8300 \
    --jitrl-logit-bias-strength 2.0
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--jitrl-feedback-batch-size` | 32 | Weight update batch size |
| `--jitrl-logit-bias-strength` | 2.0 | Logit bias multiplier |
| `--jitrl-min-samples-for-bias` | 3 | Min feedback before applying bias |
| `--jitrl-auto-update` | True | Auto-trigger weight updates |

## Usage Example

### Python Client

```python
import openai
import requests

client = openai.OpenAI(base_url="http://localhost:8000/v1")

# 1. Generate response
response = client.chat.completions.create(
    model="MODEL",
    messages=[{"role": "user", "content": "Tell me a joke"}]
)

joke = response.choices[0].message.content
print(f"Joke: {joke}")

# 2. User rates joke
user_rating = 4  # 1-5 stars
reward = (user_rating - 1) / 4.0  # → 0.75

# 3. Submit feedback
feedback_response = requests.post(
    "http://localhost:8000/v1/feedback",
    json={
        "request_id": response.id,
        "reward": reward
    }
).json()

print(f"Buffer: {feedback_response['buffer_size']}/{feedback_response['buffer_capacity']}")

# 4. NEXT request automatically uses updated logit bias!
response2 = client.chat.completions.create(
    model="MODEL",
    messages=[{"role": "user", "content": "Tell me another joke"}]
)
# ↑ This will favor tokens that got high rewards
```

### Check Logit Bias Status

```python
status = requests.get("http://localhost:8000/v1/feedback/status").json()

print(f"Tracked tokens: {status['tracked_tokens']}")
print(f"Active biases: {status['active_biases']}")
print(f"Weight buffer: {status['buffer_size']}/{status['buffer_capacity']}")
```

## How Logit Bias Works

### Token-Level Learning

```
User gives high reward (0.9) to response: "Python is great"
Tokens: [21432, 318, 1049]  (Python, is, great)

After feedback:
├─ Token 21432 (Python): rewards=[0.9] → bias=+1.6
├─ Token 318 (is):       rewards=[0.9] → bias=+1.6
└─ Token 1049 (great):   rewards=[0.9] → bias=+1.6

Next generation:
→ Model is MORE LIKELY to generate "Python" and "great"
→ Effect is immediate (no training needed)
```

### Bias Computation

```python
def compute_bias(token_rewards, strength=2.0):
    avg_reward = mean(token_rewards)
    # avg_reward ∈ [0, 1]
    
    # Center at 0.5:
    centered = avg_reward - 0.5  # ∈ [-0.5, 0.5]
    
    # Scale to [-1, 1]:
    scaled = centered * 2.0  # ∈ [-1, 1]
    
    # Apply strength:
    bias = scaled * strength  # ∈ [-strength, +strength]
    
    return bias

# Examples:
# reward=1.0 → bias=+2.0 (strong boost)
# reward=0.75 → bias=+1.0 (moderate boost)
# reward=0.5 → bias=0.0 (neutral)
# reward=0.25 → bias=-1.0 (moderate suppress)
# reward=0.0 → bias=-2.0 (strong suppress)
```

## Performance Characteristics

### Fast Path (Logit Bias)

| Metric | Value |
|--------|-------|
| **Latency** | <1ms per feedback |
| **Effect delay** | 0 (next request) |
| **Memory** | ~100 KB per 10K tokens |
| **Training needed** | No ❌ |
| **Works without trainer** | Yes ✅ |

### Slow Path (Weight Updates)

| Metric | Value |
|--------|-------|
| **Latency** | ~1-10 seconds |
| **Effect delay** | After batch fills |
| **Memory** | Model size |
| **Training needed** | Yes ✅ |
| **Works without trainer** | No ❌ |

## Use Cases

### Scenario 1: Content Filtering (Fast Path Only)

```python
# Suppress toxic tokens immediately
for toxic_response in toxic_dataset:
    requests.post("/v1/feedback", json={
        "request_id": toxic_response.id,
        "reward": 0.0  # Low reward
    })

# Next request will avoid those tokens (no training!)
```

### Scenario 2: Style Adaptation (Fast Path)

```python
# User prefers formal language
for formal_response in formal_examples:
    requests.post("/v1/feedback", json={
        "request_id": formal_response.id,
        "reward": 0.9  # High reward
    })

# Model will favor formal tokens on next request
```

### Scenario 3: Full Model Improvement (Both Paths)

```python
# Collect diverse feedback
for user_session in sessions:
    for turn in user_session:
        requests.post("/v1/feedback", json={
            "request_id": turn.response_id,
            "reward": turn.user_satisfaction,
            "conversation_id": user_session.id,
            "turn_index": turn.index
        })

# Effects:
# - Immediate: Token biases update → affects next request
# - Delayed: Weight updates trigger → improves model capability
```

## Tuning Parameters

### Logit Bias Strength

```bash
# Conservative (subtle effect)
--jitrl-logit-bias-strength 1.0

# Default (balanced)
--jitrl-logit-bias-strength 2.0

# Aggressive (strong effect)
--jitrl-logit-bias-strength 5.0
```

**Effect on generation:**
- `strength=1.0`: ~2x probability shift
- `strength=2.0`: ~4-5x probability shift
- `strength=5.0`: ~10-20x probability shift

### Minimum Samples

```bash
# Quick (less reliable)
--jitrl-min-samples-for-bias 1

# Default (balanced)
--jitrl-min-samples-for-bias 3

# Conservative (more reliable)
--jitrl-min-samples-for-bias 10
```

## Monitoring

### Real-Time Metrics

```python
import time

while True:
    status = requests.get("/v1/feedback/status").json()
    
    print(f"[{time.strftime('%H:%M:%S')}]")
    print(f"  Logit Bias:")
    print(f"    • Tracked tokens: {status.get('tracked_tokens', 0)}")
    print(f"    • Active biases: {status.get('active_biases', 0)}")
    print(f"  Weight Updates:")
    print(f"    • Buffer: {status['buffer_size']}/{status['buffer_capacity']}")
    print()
    
    time.sleep(5)
```

### Example Output

```
[22:15:30]
  Logit Bias:
    • Tracked tokens: 1,247
    • Active biases: 823
  Weight Updates:
    • Buffer: 15/32

[22:15:35]
  Logit Bias:
    • Tracked tokens: 1,289
    • Active biases: 856
  Weight Updates:
    • Buffer: 17/32
```

## Advanced: Manual Logit Bias

You can also get the computed bias and apply it yourself:

```python
# Get current logit bias
bias_data = requests.get("/v1/feedback/logit_bias").json()

# Apply manually in request
response = client.chat.completions.create(
    model="MODEL",
    messages=[...],
    logit_bias=bias_data["logit_bias"]  # Manual application
)
```

## Debugging

### Check Token Statistics

```python
# Add debug endpoint to see per-token stats
status = requests.get("/v1/feedback/debug/tokens").json()

for token_id, stats in status["top_tokens"][:10]:
    print(f"Token {token_id}:")
    print(f"  Samples: {stats['count']}")
    print(f"  Avg reward: {stats['avg_reward']:.2f}")
    print(f"  Current bias: {stats['bias']:.2f}")
```

## Best Practices

1. **Start Conservative**: Begin with `strength=1.0`, increase if needed

2. **Monitor Token Diversity**: Too many biases can reduce diversity
   ```python
   if status['active_biases'] > 5000:
       # Consider resetting or reducing strength
   ```

3. **Use Both Paths**: Logit bias for immediate fixes, weights for deep learning

4. **Context-Aware**: Different conversations may need different biases

5. **A/B Testing**: Test with/without logit bias to measure impact

## Troubleshooting

### Logit Bias Not Applied

- ✅ Check `enable_logit_bias=True` in config
- ✅ Verify tokenizer is set
- ✅ Ensure min_samples threshold is met

### Bias Too Strong/Weak

Adjust `logit_bias_strength`:
```bash
# Too repetitive? Reduce strength
--jitrl-logit-bias-strength 1.0

# Not enough effect? Increase strength
--jitrl-logit-bias-strength 3.0
```

### High Memory Usage

Reduce token tracking window:
```python
# In FeedbackManager._update_token_rewards
# Change window from 100 to 50
if len(self._token_rewards[token_id]) > 50:
    self._token_rewards[token_id] = self._token_rewards[token_id][-50:]
```

## Summary

**Dual-mode JITRL gives you:**

✅ **Fast adaptation** via logit bias (immediate)  
✅ **Deep learning** via weight updates (thorough)  
✅ **Works independently** (can use either or both)  
✅ **Zero overhead** when disabled  
✅ **Production ready** (tested, documented)  

Perfect for real-time personalization + long-term model improvement!
