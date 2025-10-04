from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Player:
    """Represents a player in the game."""
    player_id: int
    name: str = "Unknown"
    current_position: Tuple[int, int] = (0, 0)
    position_history: List[Tuple[int, int]] = field(default_factory=list)
    speed: float = 0.0  # in m/s

@dataclass
class Shuttlecock:
    """Represents the shuttlecock."""
    current_position: Tuple[int, int] = (0, 0)
    position_history: List[Tuple[int, int]] = field(default_factory=list)
    trajectory: object = None # Could be a function or a set of parameters

@dataclass
class Stroke:
    """Represents a single stroke event."""
    stroke_id: int
    player_id: int
    stroke_type: str # e.g., "Smash", "Drop", "Clear"
    start_time: float # Video timestamp
    end_time: float

@dataclass
class MatchState:
    """Represents the complete state of the match at a point in time."""
    frame_number: int
    timestamp: float
    player1: Player
    player2: Player
    shuttlecock: Shuttlecock
    score: Tuple[int, int] = (0, 0)
    last_stroke: Stroke = None
