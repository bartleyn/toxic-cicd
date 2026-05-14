from pydantic import BaseModel, Field


class ItemResult(BaseModel):
    label: int = Field(..., description="Predicted class label (0 or 1)")
    scores: dict[str, float] = Field(..., description="Score from each model, keyed by model name")
    details: dict[str, list[str]] = Field(default_factory=dict, description="Entity types detected per signal")
