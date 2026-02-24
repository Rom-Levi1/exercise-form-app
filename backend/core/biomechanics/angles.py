import math
from typing import Optional


def calculateAngle(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> Optional[float]:
    """
    Returns the angle ABC in degrees (0..180), or None if it cannot be computed.
    B is the vertex point.
    """

    # Vectors BA and BC
    baX = ax - bx
    baY = ay - by
    bcX = cx - bx
    bcY = cy - by

    magBA = math.sqrt(baX * baX + baY * baY)
    magBC = math.sqrt(bcX * bcX + bcY * bcY)

    if magBA == 0 or magBC == 0:
        return None

    dotProduct = (baX * bcX) + (baY * bcY)
    cosTheta = dotProduct / (magBA * magBC)

    # Clamp for floating-point safety
    cosTheta = max(-1.0, min(1.0, cosTheta))

    angleRad = math.acos(cosTheta)
    angleDeg = math.degrees(angleRad)

    return angleDeg


def calculateAngleFromPoints(pointA, pointB, pointC) -> Optional[float]:
    """
    pointA / pointB / pointC are expected to have .x and .y attributes
    (e.g., PosePoint objects from landmark_schema.py)
    Returns angle ABC in degrees.
    """
    if pointA is None or pointB is None or pointC is None:
        return None

    return calculateAngle(
        pointA.x, pointA.y,
        pointB.x, pointB.y,
        pointC.x, pointC.y,
    )


def getLandmarkAngle(landmarks: dict, nameA: str, nameB: str, nameC: str) -> Optional[float]:
    """
    Convenience helper: gets 3 landmarks by name from a landmarks dict and returns angle ABC.
    Example:
      getLandmarkAngle(frame.landmarks, "hip", "knee", "ankle")
    """
    pointA = landmarks.get(nameA)
    pointB = landmarks.get(nameB)
    pointC = landmarks.get(nameC)

    return calculateAngleFromPoints(pointA, pointB, pointC)