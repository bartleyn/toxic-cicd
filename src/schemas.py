from pydantic import BaseModel, Field


class ItemResult(BaseModel):
    label: int = Field(..., description="Predicted class label (0 or 1)")
    scores: dict[str, float] = Field(..., description="Score from each model, keyed by model name")
