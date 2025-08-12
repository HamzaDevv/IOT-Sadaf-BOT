# memory.pydantic_model.py
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EntityRelation(BaseModel):
    entity1: str
    relation: str
    entity2: str


class Context(BaseModel):
    """
    Pydantic model for conversation context and memory storage.
    Designed for reliable long-term memory extraction without hallucination.
    """

    summary: str = Field(
        ...,
        description=(
            "A concise, fact-based summary of the conversation segment. "
            "Must include only information explicitly stated by the user. "
            "Do not infer, assume, or fabricate details. Use clear, neutral language."
        ),
    )

    entity_relations: Optional[List[EntityRelation]] = Field(
        default=None,
        description=(
            "List of explicit factual triples in the format (entity1, relation, entity2). "
            "Only include triples directly stated by the user or confirmed in conversation. "
            "Do not guess or invent relationships."
        ),
    )

    experiential_facts: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of factual, time-bound experiences or events explicitly mentioned. "
            "Each item must be semantically distinct — no duplicate or reworded facts."
        ),
    )

    personal_facts: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of explicit personal facts about the user. "
            "Each item must be semantically distinct — no duplicate or reworded facts."
        ),
    )

    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO 8601 timestamp of when the context object was created.",
    )
