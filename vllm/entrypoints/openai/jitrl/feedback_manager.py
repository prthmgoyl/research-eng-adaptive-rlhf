# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Feedback collection and weight update coordination for JITRL.

Supports dual-mode updates:
1. Fast logit bias updates (immediate, token-level)
2. Slow weight updates (batched, model-level)
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    
    request_id: str
    conversation_id: Optional[str]
    turn_index: Optional[int]
    reward: float
    feedback_text: Optional[str]
    response_text: Optional[str]  # Captured from the actual response
    response_tokens: Optional[list[int]] = None  # Token IDs for logit updates
    timestamp: float = field(default_factory=time.time)


class FeedbackManager:
    """Manages feedback collection and triggers weight updates.
    
    Dual-mode operation:
    - Fast: Updates token-level logit biases immediately
    - Slow: Buffers feedback for batched weight updates
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        trainer_url: Optional[str] = None,
        auto_update: bool = True,
        enable_logit_bias: bool = True,
        logit_bias_strength: float = 2.0,
        min_samples_for_bias: int = 3,
    ):
        """
        Args:
            batch_size: Number of feedback entries to accumulate before weight update
            trainer_url: URL of the JITRL trainer server (e.g., http://localhost:8300)
            auto_update: Whether to automatically trigger weight updates when batch is full
            enable_logit_bias: Enable fast logit bias updates
            logit_bias_strength: Multiplier for logit bias (-strength to +strength)
            min_samples_for_bias: Minimum feedback samples before applying bias
        """
        self.batch_size = batch_size
        self.trainer_url = trainer_url
        self.auto_update = auto_update
        self.enable_logit_bias = enable_logit_bias
        self.logit_bias_strength = logit_bias_strength
        self.min_samples_for_bias = min_samples_for_bias
        
        # Weight update path (slow, batched)
        self._buffer: list[FeedbackEntry] = []
        
        # Logit bias path (fast, immediate)
        self._token_rewards: dict[int, list[float]] = defaultdict(list)
        self._logit_bias_cache: dict[str, float] = {}
        self._bias_cache_dirty = True
        
        self._lock = asyncio.Lock()
        self._response_cache: dict[str, tuple[str, Optional[list[int]]]] = {}  # request_id -> (text, tokens)
        self._trainer_client: Optional[httpx.AsyncClient] = None
        self._tokenizer = None  # Set externally if token-level tracking needed
        
        if trainer_url:
            self._trainer_client = httpx.AsyncClient(timeout=300.0)
            logger.info(
                f"JITRL FeedbackManager initialized: trainer={trainer_url}, "
                f"batch_size={batch_size}, logit_bias={enable_logit_bias}"
            )
        else:
            logger.info(
                f"JITRL FeedbackManager initialized: no trainer (feedback collection only), "
                f"logit_bias={enable_logit_bias}"
            )
    
    def set_tokenizer(self, tokenizer) -> None:
        """Set tokenizer for token-level logit bias tracking."""
        self._tokenizer = tokenizer
        logger.info("Tokenizer set for token-level feedback tracking")
    
    async def cache_response(
        self, 
        request_id: str, 
        response_text: str,
        response_tokens: Optional[list[int]] = None
    ) -> None:
        """Cache a response for later feedback association.
        
        Args:
            request_id: Request ID
            response_text: Generated text
            response_tokens: Optional pre-computed token IDs
        """
        async with self._lock:
            # Tokenize if not provided and tokenizer available
            if response_tokens is None and self._tokenizer and self.enable_logit_bias:
                try:
                    response_tokens = self._tokenizer.encode(response_text)
                except Exception as e:
                    logger.warning(f"Failed to tokenize response: {e}")
                    response_tokens = None
            
            self._response_cache[request_id] = (response_text, response_tokens)
            
            # Limit cache size (keep last 1000 responses)
            if len(self._response_cache) > 1000:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._response_cache.keys(),
                    key=lambda k: k,
                )
                for k in sorted_keys[:len(self._response_cache) - 1000]:
                    del self._response_cache[k]
    
    async def submit_feedback(
        self,
        request_id: str,
        reward: float,
        conversation_id: Optional[str] = None,
        turn_index: Optional[int] = None,
        feedback_text: Optional[str] = None,
    ) -> tuple[int, bool]:
        """
        Submit feedback for a request.
        
        Dual-mode operation:
        1. Immediately updates token-level logit biases (fast path)
        2. Buffers for batched weight updates (slow path)
        
        Returns:
            (buffer_size, update_triggered): Current buffer size and whether weight update was triggered
        """
        async with self._lock:
            # Get cached response text and tokens if available
            cached_data = self._response_cache.get(request_id)
            response_text = None
            response_tokens = None
            
            if cached_data:
                response_text, response_tokens = cached_data
            
            # FAST PATH: Update logit biases immediately
            if self.enable_logit_bias and response_tokens:
                self._update_token_rewards(response_tokens, reward)
                self._bias_cache_dirty = True
                logger.debug(
                    f"Updated logit bias for {len(response_tokens)} tokens "
                    f"with reward={reward:.2f}"
                )
            
            # SLOW PATH: Buffer for weight updates
            entry = FeedbackEntry(
                request_id=request_id,
                conversation_id=conversation_id,
                turn_index=turn_index,
                reward=reward,
                feedback_text=feedback_text,
                response_text=response_text,
                response_tokens=response_tokens,
            )
            
            self._buffer.append(entry)
            current_size = len(self._buffer)
            
            logger.info(
                f"Feedback received: request_id={request_id}, reward={reward:.2f}, "
                f"buffer={current_size}/{self.batch_size}"
            )
        
        # Check if we should trigger update
        update_triggered = False
        if self.auto_update and current_size >= self.batch_size:
            # Trigger update in background
            asyncio.create_task(self._trigger_update())
            update_triggered = True
        
        return current_size, update_triggered
    
    def _update_token_rewards(self, tokens: list[int], reward: float) -> None:
        """Update token-level reward statistics (called with lock held).
        
        Args:
            tokens: List of token IDs from the response
            reward: Reward score for this response
        """
        for token_id in tokens:
            self._token_rewards[token_id].append(reward)
            
            # Keep only recent rewards (window of 100 per token)
            if len(self._token_rewards[token_id]) > 100:
                self._token_rewards[token_id] = self._token_rewards[token_id][-100:]
    
    def _compute_logit_bias(self) -> dict[str, float]:
        """Compute current logit bias from token reward statistics.
        
        Returns:
            Dictionary mapping token_id (as string) to bias value
        """
        if not self.enable_logit_bias or not self._token_rewards:
            return {}
        
        bias = {}
        
        for token_id, rewards in self._token_rewards.items():
            # Need minimum samples for reliable bias
            if len(rewards) < self.min_samples_for_bias:
                continue
            
            # Compute average reward
            avg_reward = sum(rewards) / len(rewards)
            
            # Map reward (0-1) to bias (-strength to +strength)
            # reward=0.5 → bias=0 (neutral)
            # reward=1.0 → bias=+strength (boost)
            # reward=0.0 → bias=-strength (suppress)
            bias_value = (avg_reward - 0.5) * 2.0 * self.logit_bias_strength
            
            # Only include non-zero biases
            if abs(bias_value) > 0.01:
                bias[str(token_id)] = bias_value
        
        return bias
    
    async def get_logit_bias(self) -> dict[str, float]:
        """Get current logit bias for generation.
        
        Returns immediately with cached bias, recomputes if dirty.
        This is called on EVERY request, so must be fast.
        """
        async with self._lock:
            if self._bias_cache_dirty:
                self._logit_bias_cache = self._compute_logit_bias()
                self._bias_cache_dirty = False
                logger.debug(
                    f"Recomputed logit bias: {len(self._logit_bias_cache)} tokens"
                )
        
        return self._logit_bias_cache
    
    async def get_buffer_status(self) -> dict[str, Any]:
        """Get current buffer status including logit bias info."""
        async with self._lock:
            logit_bias_status = {}
            if self.enable_logit_bias:
                logit_bias_status = {
                    "logit_bias_enabled": True,
                    "tracked_tokens": len(self._token_rewards),
                    "active_biases": len(self._logit_bias_cache),
                    "bias_strength": self.logit_bias_strength,
                }
            
            return {
                "buffer_size": len(self._buffer),
                "buffer_capacity": self.batch_size,
                "trainer_url": self.trainer_url,
                "auto_update": self.auto_update,
                **logit_bias_status,
            }
    
    async def _trigger_update(self) -> None:
        """Trigger a weight update with the current feedback batch."""
        async with self._lock:
            if len(self._buffer) == 0:
                logger.info("No feedback to send")
                return
            
            feedback_batch = list(self._buffer)
            self._buffer.clear()
        
        if not self.trainer_url or not self._trainer_client:
            logger.warning(
                f"No trainer URL configured, discarding {len(feedback_batch)} feedback entries"
            )
            return
        
        logger.info(f"Triggering weight update with {len(feedback_batch)} feedback entries")
        
        try:
            # Convert to JSON format
            feedback_json = [
                {
                    "request_id": f.request_id,
                    "conversation_id": f.conversation_id,
                    "turn_index": f.turn_index,
                    "reward": f.reward,
                    "feedback_text": f.feedback_text,
                    "response_text": f.response_text,
                    "timestamp": f.timestamp,
                }
                for f in feedback_batch
            ]
            
            # Send to trainer
            response = await self._trainer_client.post(
                f"{self.trainer_url}/train",
                json={"feedback": feedback_json},
            )
            response.raise_for_status()
            
            logger.info(f"Successfully sent {len(feedback_batch)} feedback entries to trainer")
            
        except Exception as e:
            logger.error(f"Failed to send feedback to trainer: {e}", exc_info=True)
            # Re-add feedback to buffer on failure
            async with self._lock:
                self._buffer.extend(feedback_batch)
    
    async def manual_update(self) -> dict[str, Any]:
        """Manually trigger a weight update regardless of buffer size."""
        async with self._lock:
            buffer_size = len(self._buffer)
        
        if buffer_size == 0:
            return {
                "status": "no_feedback",
                "message": "No feedback to process",
            }
        
        await self._trigger_update()
        
        return {
            "status": "update_triggered",
            "feedback_count": buffer_size,
        }
    
    async def shutdown(self) -> None:
        """Shutdown the feedback manager."""
        if self._trainer_client:
            await self._trainer_client.aclose()
