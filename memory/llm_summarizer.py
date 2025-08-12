# memory.llm_summarizer.py
from typing import List, Tuple
import logging
from memory.pydantic_model import Context
from config import GLOBAL_LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationSummarizer:
    """Summarizes conversations into the strict Context schema."""

    def __init__(self):
        self.llm = GLOBAL_LLM.with_structured_output(Context)

    def _build_prompt(self, conversation_data: List[Tuple[str, str]]) -> str:
        conv_text = "\n".join(
            f"Turn {i + 1}:\nUser: {u}\nAI: {a}\n"
            for i, (u, a) in enumerate(conversation_data)
        )
        return f"""
You are an expert conversation analyst. 
Extract ONLY explicitly stated facts from the conversation into the Context JSON schema.

Rules for fact extraction:
1. Facts must be explicit in the conversation — no guessing or inference.
2. Facts in `experiential_facts` and `personal_facts` must be provided as **lists of bullet points**.
3. Each fact in these lists must be **semantically distinct** — avoid reworded duplicates.
4. If no facts exist for a field, return null.
5. Keep facts short, clear, and neutral.

CONVERSATION:
{conv_text}

OUTPUT FORMAT:
{{
    "summary": "short fact-based summary",
    "entity_relations": [
        {{
            "entity1": "...",
            "relation": "...",
            "entity2": "..."
        }},
        ...
    ] OR null,
    "experiential_facts": ["fact1", "fact2", ...] OR null,
    "personal_facts": ["fact1", "fact2", ...] OR null,
    "timestamp": "ISO timestamp"
}}
"""

    def summarize_conversation(
        self, conversation_data: List[Tuple[str, str]]
    ) -> Context:
        """Returns structured conversation summary with semantic deduplication."""
        try:
            prompt = self._build_prompt(conversation_data)
            ctx: Context = self.llm.invoke(prompt)
            with open("conversation_summary.txt", "a", encoding="utf-8") as f:
                f.write(ctx.model_dump_json(indent=2) + "\n\n")
            return ctx

        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return Context(
                summary="Error occurred during summarization",
                entity_relations=None,
                experiential_facts=None,
                personal_facts=None,
            )
