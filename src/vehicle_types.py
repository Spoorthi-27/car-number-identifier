"""
vehicle_type.py
---------------
Single place that turns a plate color into a vehicle classification.

    WHITE   -> Private Vehicle
    YELLOW  -> Commercial / Transport Vehicle
    GREEN   -> Electric Vehicle
    other   -> Unknown

Nothing further (cab, owned car, fuel type, ...) is inferred.
"""
from typing import Optional

UNKNOWN_VEHICLE_TYPE = "Unknown"

VEHICLE_TYPE_BY_COLOR = {
    "WHITE": "Private Vehicle",
    "YELLOW": "Commercial / Transport Vehicle",
    "GREEN": "Electric Vehicle",
}


def classify_vehicle_type(plate_color: Optional[str]) -> str:
    """Return the vehicle type for a plate color; 'Unknown' for anything else."""
    key = (plate_color or "").strip().upper()
    return VEHICLE_TYPE_BY_COLOR.get(key, UNKNOWN_VEHICLE_TYPE)