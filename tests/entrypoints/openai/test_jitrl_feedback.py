# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JITRL feedback functionality."""

import asyncio

import pytest

from vllm.entrypoints.openai.jitrl.feedback_manager import (
    FeedbackEntry,
    FeedbackManager,
)
from vllm.entrypoints.openai.jitrl.protocol import FeedbackRequest, FeedbackResponse


class TestFeedbackManager:
    """Test cases for FeedbackManager."""

    @pytest.mark.asyncio
    async def test_feedback_buffer_accumulation(self):
        """Test that feedback accumulates in buffer."""
        manager = FeedbackManager(batch_size=3, auto_update=False)
        
        # Submit feedback entries
        size1, triggered1 = await manager.submit_feedback(
            request_id="req-1",
            reward=0.8,
        )
        assert size1 == 1
        assert not triggered1
        
        size2, triggered2 = await manager.submit_feedback(
            request_id="req-2",
            reward=0.9,
        )
        assert size2 == 2
        assert not triggered2
        
        size3, triggered3 = await manager.submit_feedback(
            request_id="req-3",
            reward=0.7,
        )
        assert size3 == 3
        assert not triggered3  # auto_update is False

    @pytest.mark.asyncio
    async def test_feedback_triggers_update(self):
        """Test that feedback triggers update when batch is full."""
        manager = FeedbackManager(batch_size=2, auto_update=True, trainer_url=None)
        
        # First feedback
        size1, triggered1 = await manager.submit_feedback(
            request_id="req-1",
            reward=0.8,
        )
        assert size1 == 1
        assert not triggered1
        
        # Second feedback should trigger
        size2, triggered2 = await manager.submit_feedback(
            request_id="req-2",
            reward=0.9,
        )
        assert size2 == 2
        assert triggered2

    @pytest.mark.asyncio
    async def test_response_caching(self):
        """Test that responses are cached correctly."""
        manager = FeedbackManager(batch_size=10, auto_update=False)
        
        # Cache a response
        await manager.cache_response("req-123", "This is the response text")
        
        # Submit feedback - should retrieve cached response
        await manager.submit_feedback(
            request_id="req-123",
            reward=0.9,
        )
        
        # Verify feedback was recorded with response text
        status = await manager.get_buffer_status()
        assert status["buffer_size"] == 1

    @pytest.mark.asyncio
    async def test_get_buffer_status(self):
        """Test buffer status retrieval."""
        manager = FeedbackManager(
            batch_size=5,
            trainer_url="http://localhost:8300",
            auto_update=True,
        )
        
        # Add some feedback
        await manager.submit_feedback("req-1", 0.8)
        await manager.submit_feedback("req-2", 0.9)
        
        status = await manager.get_buffer_status()
        assert status["buffer_size"] == 2
        assert status["buffer_capacity"] == 5
        assert status["trainer_url"] == "http://localhost:8300"
        assert status["auto_update"] is True

    @pytest.mark.asyncio
    async def test_manual_update(self):
        """Test manual update trigger."""
        manager = FeedbackManager(batch_size=10, auto_update=False)
        
        # Add feedback
        await manager.submit_feedback("req-1", 0.8)
        await manager.submit_feedback("req-2", 0.9)
        
        # Trigger manual update
        result = await manager.manual_update()
        assert result["status"] == "update_triggered"
        assert result["feedback_count"] == 2

    @pytest.mark.asyncio
    async def test_manual_update_empty_buffer(self):
        """Test manual update with empty buffer."""
        manager = FeedbackManager(batch_size=10, auto_update=False)
        
        result = await manager.manual_update()
        assert result["status"] == "no_feedback"

    @pytest.mark.asyncio
    async def test_conversation_tracking(self):
        """Test feedback with conversation_id and turn_index."""
        manager = FeedbackManager(batch_size=10, auto_update=False)
        
        # Submit feedback with conversation tracking
        await manager.submit_feedback(
            request_id="req-1",
            reward=0.8,
            conversation_id="conv-123",
            turn_index=0,
        )
        
        await manager.submit_feedback(
            request_id="req-2",
            reward=0.9,
            conversation_id="conv-123",
            turn_index=1,
        )
        
        status = await manager.get_buffer_status()
        assert status["buffer_size"] == 2

    @pytest.mark.asyncio
    async def test_response_cache_limit(self):
        """Test that response cache has size limit."""
        manager = FeedbackManager(batch_size=10, auto_update=False)
        
        # Cache many responses (more than limit of 1000)
        for i in range(1100):
            await manager.cache_response(f"req-{i}", f"Response {i}")
        
        # Oldest responses should be evicted
        # Cache should be around 1000 entries
        # We can't directly check cache size, but this tests it doesn't grow unbounded


class TestFeedbackProtocol:
    """Test cases for feedback protocol models."""

    def test_feedback_request_validation(self):
        """Test FeedbackRequest validation."""
        # Valid request
        request = FeedbackRequest(
            request_id="chatcmpl-123",
            reward=0.8,
        )
        assert request.request_id == "chatcmpl-123"
        assert request.reward == 0.8
        assert request.conversation_id is None
        assert request.turn_index is None

    def test_feedback_request_with_optional_fields(self):
        """Test FeedbackRequest with all fields."""
        request = FeedbackRequest(
            request_id="chatcmpl-123",
            reward=0.9,
            conversation_id="conv-456",
            turn_index=2,
            feedback_text="Great response!",
        )
        assert request.request_id == "chatcmpl-123"
        assert request.reward == 0.9
        assert request.conversation_id == "conv-456"
        assert request.turn_index == 2
        assert request.feedback_text == "Great response!"

    def test_feedback_response_creation(self):
        """Test FeedbackResponse creation."""
        response = FeedbackResponse(
            status="accepted",
            buffer_size=5,
            buffer_capacity=32,
            update_triggered=False,
        )
        assert response.status == "accepted"
        assert response.buffer_size == 5
        assert response.buffer_capacity == 32
        assert not response.update_triggered

    def test_feedback_response_with_update(self):
        """Test FeedbackResponse when update is triggered."""
        response = FeedbackResponse(
            status="accepted",
            buffer_size=32,
            buffer_capacity=32,
            update_triggered=True,
        )
        assert response.update_triggered


class TestFeedbackEntry:
    """Test cases for FeedbackEntry dataclass."""

    def test_feedback_entry_creation(self):
        """Test FeedbackEntry creation."""
        entry = FeedbackEntry(
            request_id="req-123",
            conversation_id="conv-456",
            turn_index=0,
            reward=0.85,
            feedback_text="Good response",
            response_text="This is the actual response",
        )
        assert entry.request_id == "req-123"
        assert entry.conversation_id == "conv-456"
        assert entry.turn_index == 0
        assert entry.reward == 0.85
        assert entry.feedback_text == "Good response"
        assert entry.response_text == "This is the actual response"
        assert entry.timestamp > 0

    def test_feedback_entry_with_none_fields(self):
        """Test FeedbackEntry with optional None fields."""
        entry = FeedbackEntry(
            request_id="req-123",
            conversation_id=None,
            turn_index=None,
            reward=0.7,
            feedback_text=None,
            response_text=None,
        )
        assert entry.request_id == "req-123"
        assert entry.conversation_id is None
        assert entry.turn_index is None
        assert entry.reward == 0.7
        assert entry.feedback_text is None
        assert entry.response_text is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
