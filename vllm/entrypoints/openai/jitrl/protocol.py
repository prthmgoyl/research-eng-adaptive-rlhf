# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol definitions for JITRL feedback."""

from typing import Optional

from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """Request to submit feedback for a chat completion response."""
    
    request_id: str = Field(
        ...,
        description="The request ID from the chat completion response",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for multi-turn tracking",
    )
    turn_index: Optional[int] = Field(
        None,
        description="Optional turn index within the conversation",
    )
    reward: float = Field(
        ...,
        description="Reward score (typically 0.0-1.0 or -1.0 to 1.0)",
    )
    feedback_text: Optional[str] = Field(
        None,
        description="Optional human-readable feedback",
    )


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    
    status: str = Field(
        ...,
        description="Status of the feedback submission",
    )
    buffer_size: int = Field(
        ...,
        description="Current size of the feedback buffer",
    )
    buffer_capacity: int = Field(
        ...,
        description="Maximum buffer size before triggering update",
    )
    update_triggered: bool = Field(
        default=False,
        description="Whether this feedback triggered a weight update",
    )
