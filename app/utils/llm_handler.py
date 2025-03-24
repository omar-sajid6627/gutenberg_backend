"""
Answers from Groq's hosted Llama when GROQ_API_KEY is set, and from the
extractive LocalLLMHandler when it is not.
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.utils.local_llm import LocalLLMHandler, usable_api_key

load_dotenv()

logger = logging.getLogger(__name__)

# `os.environ["GROQ_API_KEY"] = GROQ_API_KEY` raised
# `TypeError: str expected, not NoneType` whenever the key was unset, so this
# module — and therefore the whole API — could not even be imported without a
# Groq account. The error named neither the key nor the cause.
GROQ_API_KEY = usable_api_key("GROQ_API_KEY")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

MODEL = "llama-3.1-8b-instant"
API_TIMEOUT = 30  # seconds


def build_prompt(query: str, context: str | None) -> str:
    if not context:
        return query
    return (
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        "Provide a concise answer based on the given context. "
        "Keep your response brief and to the point."
    )


class LLMHandler:
    is_local = False

    def __init__(self, llm=None):
        if llm is None:
            from langchain_groq import ChatGroq

            llm = ChatGroq(model=MODEL, temperature=0, max_tokens=1000, timeout=API_TIMEOUT, max_retries=1)
        self.llm = llm

    async def generate_response(self, query: str, context: str | None = None, temperature: float = 0) -> str:
        messages = [HumanMessage(content=build_prompt(query, context))]
        try:
            # The request's temperature used to be accepted and then ignored.
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages, temperature=temperature), timeout=API_TIMEOUT
            )
        except TimeoutError:
            logger.warning("Groq request timed out after %ss", API_TIMEOUT)
            return "I'm sorry, but the request to the AI service timed out. Please try again later."
        except Exception:
            # Upstream errors can echo request details; log them, don't return them.
            logger.exception("Groq request failed")
            return "I'm sorry, but I encountered an error while generating a response. Please try again."
        return response.content


def _build_handler():
    """
    Pick a handler: the hosted model when a usable key is present, the local
    extractive one otherwise. A key that fails to initialise degrades to the
    local handler rather than making the API unimportable.
    """
    if not GROQ_API_KEY:
        logger.info("No GROQ_API_KEY set; answering extractively from book text.")
        return LocalLLMHandler()
    try:
        return LLMHandler()
    except Exception as exc:
        logger.warning("Groq unavailable (%s); falling back to extractive answers.", exc)
        return LocalLLMHandler()


llm_handler = _build_handler()
