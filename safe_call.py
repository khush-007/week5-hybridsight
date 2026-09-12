"""
Reusable error-handling utilities for HybridSight.
"""

import os
import functools


def safe_call(func):
    """
    Decorator for Gradio event handlers.

    Converts common application/API errors into
    user-friendly messages instead of exposing
    Python tracebacks in the UI.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        try:
            # -----------------------------------------
            # API KEY CHECK
            # -----------------------------------------

            if not os.getenv("GROQ_API_KEY"):

                return (
                    "❌ Configuration error: GROQ_API_KEY "
                    "is not configured."
                )

            # -----------------------------------------
            # NORMAL FUNCTION
            # -----------------------------------------

            return func(*args, **kwargs)

        except Exception as e:

            error_message = str(e)

            print("\n===================================")
            print("HYBRIDSIGHT ERROR")
            print("===================================")
            print(f"Function : {func.__name__}")
            print(f"Error    : {error_message}")
            print("===================================\n")

            # -----------------------------------------
            # RATE LIMIT
            # -----------------------------------------

            if (
                "rate limit" in error_message.lower()
                or "429" in error_message
            ):
                return (
                    "⚠️ API rate limit reached. "
                    "Please wait a moment and try again."
                )

            # -----------------------------------------
            # AUTHENTICATION
            # -----------------------------------------

            if (
                "401" in error_message
                or "unauthorized" in error_message.lower()
                or "invalid api key" in error_message.lower()
            ):
                return (
                    "❌ API authentication failed. "
                    "Please check your GROQ_API_KEY."
                )

            # -----------------------------------------
            # TIMEOUT / NETWORK
            # -----------------------------------------

            if (
                "timeout" in error_message.lower()
                or "timed out" in error_message.lower()
                or "connection" in error_message.lower()
                or "network" in error_message.lower()
            ):
                return (
                    "🌐 Network/API timeout. "
                    "Please check your internet connection "
                    "and try again."
                )

            # -----------------------------------------
            # GRAPH RECURSION
            # -----------------------------------------

            if "GraphRecursionError" in error_message:

                return (
                    "⚠️ The agent reached its reasoning limit. "
                    "Please try a simpler question."
                )

            # -----------------------------------------
            # EMPTY VECTOR DATABASE
            # -----------------------------------------

            if (
                "chroma" in error_message.lower()
                or "no documents" in error_message.lower()
            ):
                return (
                    "📄 No document information is currently "
                    "available. Please upload and index a PDF first."
                )

            # -----------------------------------------
            # GENERIC ERROR
            # -----------------------------------------

            return (
                "❌ Something went wrong while processing "
                "your request. Please try again."
            )

    return wrapper