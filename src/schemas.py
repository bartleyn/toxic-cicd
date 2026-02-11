from pydantic import BaseModel, Field


class ItemResult(BaseModel):
    toxicity_score: float = Field(..., description="Predicted probability of toxic class")
    label: int = Field(..., description="Predicted class label (0 or 1)")
    sentiment_score: float = Field(0.0, description="Sentiment score from -1 (negative) to 1 (positive)")
