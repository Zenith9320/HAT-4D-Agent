from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RecordedClick:
    x: int
    y: int
    label: int  # 1=positive, 0=negative
    timestamp: float


@dataclass
class RecordedObject:
    clicks: List[RecordedClick] = field(default_factory=list)
    mask_path: Optional[str] = None
    rgb_path: Optional[str] = None
    bbox_center: Optional[List[float]] = None
    bbox_dimensions: Optional[List[float]] = None
    bbox_wxyz: Optional[List[float]] = None


class RecordOpsState:
    def __init__(self) -> None:
        self.lock = threading.Lock()

        # Global toggle
        self.recording_enabled: bool = False

        # Session state
        self.session_active: bool = False
        self.session_dir: Optional[Path] = None
        self.session_id: Optional[str] = None
        self.frame_token: Optional[str] = None
        self.started_at: Optional[str] = None

        # Per-object tracking
        self.objects: List[RecordedObject] = []
        self.current_object_idx: int = 0

        # UI handles
        self.toggle_handle: Any = None
        self.status_handle: Any = None
