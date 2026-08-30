#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example client for JITRL (Just-In-Time Reinforcement Learning) with vLLM.

This script demonstrates:
1. Generating completions from a vLLM server with JITRL enabled
2. Collecting user feedback (rewards) for responses
3. Submitting feedback to trigger weight updates

Usage:
    # Start vLLM server first:
    vllm serve Qwen/Qwen3-1.7B --enable-jitrl-feedback \\
        --jitrl-trainer-url http://localhost:8300

    # Then run this script:
    python client_example.py --base-url http://localhost:8000
"""

import argparse
import random
import time

import openai
import requests


def simulate_user_reward(response_text: str) -> float:
    """
    Simulate a user providing a reward for a response.
    
    In a real system, this would be:
    - Explicit user feedback (thumbs up/down, star rating, etc.)
    - Implicit signals (click-through rate, dwell time, etc.)
    - Reward model predictions
    - Human preferences from comparison interfaces
    
    For this demo, we generate random rewards with a bias towards
    responses containing certain keywords.
    """
    # Base random reward
    reward = random.uniform(0.3, 0.9)
    
    # Boost reward for informative responses
    good_indicators = [
        "explain", "because", "however", "therefore",
        "example", "specifically", "detail",
    ]
    
    for indicator in good_indicators:
        if indicator.lower() in response_text.lower():
            reward = min(1.0, reward + 0.1)
    
    # Penalize very short responses
    if len(response_text) < 50:
        reward *= 0.7
    
    return round(reward, 2)


def submit_feedback(
    base_url: str,
    request_id: str,
    reward: float,
    conversation_id: str | None = None,
    turn_index: int | None = None,
) -> dict:
    """Submit feedback to the vLLM server."""
    feedback = {
        "request_id": request_id,
        "reward": reward,
    }
    
    if conversation_id:
        feedback["conversation_id"] = conversation_id
    if turn_index is not None:
        feedback["turn_index"] = turn_index
    
    response = requests.post(
        f"{base_url}/v1/feedback",
        json=feedback,
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="JITRL client example")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the vLLM server",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="Model name to use",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests to send",
    )
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = openai.OpenAI(
        base_url=f"{args.base_url}/v1",
        api_key="dummy",  # vLLM doesn't require a real key by default
    )
    
    # Example prompts
    prompts = [
        "Explain how photosynthesis works.",
        "What are the benefits of regular exercise?",
        "Describe the water cycle.",
        "What is machine learning?",
        "How do vaccines work?",
        "Explain the concept of supply and demand.",
        "What causes seasons to change?",
        "Describe the structure of an atom.",
        "What is climate change?",
        "How does the internet work?",
    ]
    
    print(f"Starting JITRL example with {args.num_requests} requests")
    print(f"Server: {args.base_url}")
    print(f"Model: {args.model}\n")
    
    # Track feedback statistics
    total_reward = 0.0
    feedback_submitted = 0
    
    for i in range(args.num_requests):
        prompt = prompts[i % len(prompts)]
        
        print(f"\n{'='*60}")
        print(f"Request {i+1}/{args.num_requests}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        
        # Generate response
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            
            request_id = response.id
            response_text = response.choices[0].message.content
            
            print(f"\nResponse ({request_id}):")
            print(f"{response_text}")
            
            # Simulate user feedback
            reward = simulate_user_reward(response_text)
            total_reward += reward
            
            print(f"\nSimulated reward: {reward:.2f}")
            
            # Submit feedback
            feedback_result = submit_feedback(
                args.base_url,
                request_id,
                reward,
                conversation_id=f"conv-{i // 3}",  # Group into conversations
                turn_index=i % 3,
            )
            
            feedback_submitted += 1
            
            print(f"Feedback submitted: {feedback_result}")
            
            # Check if update was triggered
            if feedback_result.get("update_triggered"):
                print("\n*** WEIGHT UPDATE TRIGGERED ***")
                print("Training will begin and weights will be synced...")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Final statistics
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total requests: {args.num_requests}")
    print(f"Feedback submitted: {feedback_submitted}")
    print(f"Average reward: {total_reward / feedback_submitted:.2f}")
    
    # Check final buffer status
    try:
        status_response = requests.get(f"{args.base_url}/v1/feedback/status")
        status = status_response.json()
        print(f"\nFinal buffer status:")
        print(f"  Buffer size: {status['buffer_size']}/{status['buffer_capacity']}")
        print(f"  Auto-update: {status['auto_update']}")
        
        if status['buffer_size'] > 0:
            print(f"\nNote: {status['buffer_size']} feedback entries are queued.")
            print("They will trigger an update when the buffer is full,")
            print("or you can manually trigger with:")
            print(f"  curl -X POST {args.base_url}/v1/feedback/trigger_update")
    except Exception as e:
        print(f"Could not get buffer status: {e}")


if __name__ == "__main__":
    main()
