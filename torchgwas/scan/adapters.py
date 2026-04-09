"""Adapters making all models look identical to the scanner."""

from __future__ import annotations

from ..models.base import BaseModel


class ModelAdapter:
    """Wraps a BaseModel to normalize calling conventions for the scanner."""

    def __init__(self, model: BaseModel) -> None:
        self.model = model
