# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for JITRL feedback endpoints."""

import pytest
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.jitrl.feedback_manager import FeedbackManager


@pytest.fixture
def mock_app_with_feedback():
    """Create a mock FastAPI app with feedback manager."""
    from fastapi import FastAPI
    from vllm.entrypoints.openai.jitrl.api_router import attach_router
    
    app = FastAPI()
    
    # Initialize feedback manager
    app.state.feedback_manager = FeedbackManager(
        batch_size=3,
        trainer_url=None,
        auto_update=False,
    )
    
    # Attach JITRL routes
    attach_router(app)
    
    return app


@pytest.fixture
def client(mock_app_with_feedback):
    """Create test client."""
    return TestClient(mock_app_with_feedback)


class TestFeedbackEndpoints:
    """Test JITRL feedback API endpoints."""

    def test_submit_feedback_success(self, client):
        """Test successful feedback submission."""
        response = client.post(
            "/v1/feedback",
            json={
                "request_id": "chatcmpl-123",
                "reward": 0.85,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert data["buffer_size"] == 1
        assert data["buffer_capacity"] == 3
        assert data["update_triggered"] is False

    def test_submit_feedback_with_all_fields(self, client):
        """Test feedback submission with all optional fields."""
        response = client.post(
            "/v1/feedback",
            json={
                "request_id": "chatcmpl-456",
                "reward": 0.9,
                "conversation_id": "conv-789",
                "turn_index": 2,
                "feedback_text": "Excellent response!",
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"

    def test_submit_feedback_missing_required_field(self, client):
        """Test feedback submission with missing required field."""
        response = client.post(
            "/v1/feedback",
            json={
                "request_id": "chatcmpl-123",
                # Missing 'reward'
            },
        )
        
        assert response.status_code == 422  # Validation error

    def test_get_feedback_status(self, client):
        """Test getting feedback buffer status."""
        # Submit some feedback first
        client.post(
            "/v1/feedback",
            json={"request_id": "req-1", "reward": 0.8},
        )
        client.post(
            "/v1/feedback",
            json={"request_id": "req-2", "reward": 0.9},
        )
        
        # Get status
        response = client.get("/v1/feedback/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["buffer_size"] == 2
        assert data["buffer_capacity"] == 3
        assert data["auto_update"] is False

    def test_trigger_manual_update(self, client):
        """Test manually triggering weight update."""
        # Submit feedback
        client.post(
            "/v1/feedback",
            json={"request_id": "req-1", "reward": 0.8},
        )
        
        # Trigger update
        response = client.post("/v1/feedback/trigger_update")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "update_triggered"
        assert data["feedback_count"] == 1

    def test_trigger_update_empty_buffer(self, client):
        """Test triggering update with empty buffer."""
        response = client.post("/v1/feedback/trigger_update")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_feedback"

    def test_multiple_feedback_submissions(self, client):
        """Test submitting multiple feedback entries."""
        # Submit multiple feedback
        for i in range(5):
            response = client.post(
                "/v1/feedback",
                json={
                    "request_id": f"req-{i}",
                    "reward": 0.7 + (i * 0.05),
                },
            )
            assert response.status_code == 200
        
        # Check final status
        status_response = client.get("/v1/feedback/status")
        data = status_response.json()
        assert data["buffer_size"] == 5


@pytest.fixture
def mock_app_without_feedback():
    """Create a mock FastAPI app without feedback manager."""
    from fastapi import FastAPI
    from vllm.entrypoints.openai.jitrl.api_router import attach_router
    
    app = FastAPI()
    # Don't add feedback_manager to state
    attach_router(app)
    
    return app


@pytest.fixture
def client_no_feedback(mock_app_without_feedback):
    """Create test client without feedback enabled."""
    return TestClient(mock_app_without_feedback)


class TestFeedbackDisabled:
    """Test behavior when JITRL is not enabled."""

    def test_submit_feedback_when_disabled(self, client_no_feedback):
        """Test feedback submission when JITRL is disabled."""
        response = client_no_feedback.post(
            "/v1/feedback",
            json={
                "request_id": "chatcmpl-123",
                "reward": 0.85,
            },
        )
        
        assert response.status_code == 503  # Service unavailable
        data = response.json()
        assert "not enabled" in data["detail"].lower()

    def test_get_status_when_disabled(self, client_no_feedback):
        """Test getting status when JITRL is disabled."""
        response = client_no_feedback.get("/v1/feedback/status")
        
        assert response.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
