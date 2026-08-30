# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API router for JITRL feedback endpoints."""

from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.jitrl.feedback_manager import FeedbackManager
from vllm.entrypoints.openai.jitrl.protocol import FeedbackRequest, FeedbackResponse
from vllm.entrypoints.serve.utils.api_utils import validate_json_request
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def get_feedback_manager(request: Request) -> FeedbackManager:
    """Get the feedback manager from app state."""
    manager = getattr(request.app.state, "feedback_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="JITRL feedback is not enabled on this server",
        )
    return manager


@router.post(
    "/v1/feedback",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": FeedbackResponse},
        HTTPStatus.BAD_REQUEST.value: {},
        HTTPStatus.SERVICE_UNAVAILABLE.value: {},
    },
)
async def submit_feedback(
    feedback: FeedbackRequest,
    raw_request: Request,
):
    """
    Submit feedback for a chat completion response.
    
    This endpoint allows clients to provide reward signals for reinforcement learning.
    When enough feedback is collected, it automatically triggers a weight update.
    
    Example:
        ```
        POST /v1/feedback
        {
            "request_id": "chatcmpl-abc123",
            "conversation_id": "conv-456",
            "turn_index": 0,
            "reward": 0.8
        }
        ```
    """
    manager = get_feedback_manager(raw_request)
    
    buffer_size, update_triggered = await manager.submit_feedback(
        request_id=feedback.request_id,
        reward=feedback.reward,
        conversation_id=feedback.conversation_id,
        turn_index=feedback.turn_index,
        feedback_text=feedback.feedback_text,
    )
    
    return JSONResponse(
        content=FeedbackResponse(
            status="accepted",
            buffer_size=buffer_size,
            buffer_capacity=manager.batch_size,
            update_triggered=update_triggered,
        ).model_dump()
    )


@router.get("/v1/feedback/status")
async def get_feedback_status(raw_request: Request):
    """
    Get the current status of the feedback buffer.
    
    Returns information about how many feedback entries are queued
    and whether auto-updates are enabled.
    """
    manager = get_feedback_manager(raw_request)
    status = await manager.get_buffer_status()
    return JSONResponse(content=status)


@router.post("/v1/feedback/trigger_update")
async def trigger_manual_update(raw_request: Request):
    """
    Manually trigger a weight update regardless of buffer size.
    
    This is useful for testing or when you want immediate updates
    without waiting for the buffer to fill.
    """
    manager = get_feedback_manager(raw_request)
    result = await manager.manual_update()
    return JSONResponse(content=result)


def attach_router(app: FastAPI):
    """Attach the JITRL feedback router to the FastAPI app."""
    app.include_router(router)
