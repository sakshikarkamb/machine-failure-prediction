from pydantic import BaseModel, Field
from typing import Literal

class PredictionInput(BaseModel):
    type: Literal["L", "M", "H"] = Field(
        ...,
        description="Machine product type: L (Low quality - 50%), M (Medium quality - 30%), H (High quality - 20%)",
        examples=["M"]
    )
    air_temperature: float = Field(
        ...,
        description="Air temperature in Kelvin [K]",
        ge=200.0,
        le=400.0,
        examples=[300.0]
    )
    process_temperature: float = Field(
        ...,
        description="Process temperature in Kelvin [K]",
        ge=200.0,
        le=400.0,
        examples=[310.0]
    )
    rotational_speed: float = Field(
        ...,
        description="Rotational speed in revolutions per minute [rpm]",
        ge=100.0,
        le=5000.0,
        examples=[1500.0]
    )
    torque: float = Field(
        ...,
        description="Torque in Newton meters [Nm]",
        ge=0.0,
        le=200.0,
        examples=[40.0]
    )
    tool_wear: float = Field(
        ...,
        description="Tool wear time in minutes [min]",
        ge=0.0,
        le=500.0,
        examples=[100.0]
    )

class PredictionResponse(BaseModel):
    prediction: int = Field(
        ...,
        description="Predicted class: 0 for No Failure, 1 for Machine Failure",
        examples=[0]
    )
    result: str = Field(
        ...,
        description="Human-readable result: 'No Machine Failure' or 'Machine Failure'",
        examples=["No Machine Failure"]
    )
    failure_probability: float = Field(
        ...,
        description="Probability of machine failure (between 0.0 and 1.0)",
        examples=[0.15]
    )

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])

class RootResponse(BaseModel):
    message: str = Field(..., examples=["Machine Failure Prediction API is running"])
