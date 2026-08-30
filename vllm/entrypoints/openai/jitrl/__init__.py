# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Just-In-Time Reinforcement Learning (JITRL) support for vLLM.

Provides feedback collection and weight update coordination for online RL.
"""

from vllm.entrypoints.openai.jitrl.feedback_manager import FeedbackManager
from vllm.entrypoints.openai.jitrl.protocol import FeedbackRequest, FeedbackResponse

__all__ = ["FeedbackManager", "FeedbackRequest", "FeedbackResponse"]
