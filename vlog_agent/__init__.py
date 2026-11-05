"""vlog agent 骨架：Project 状态机与素材技术判定。"""

from .schema import (
    DetectorResult,
    GateEvent,
    GateType,
    IntentCard,
    MaterialCard,
    ProjectState,
    Shot,
    ShotStatus,
    Stage,
    StateEvent,
    TechnicalJudgement,
    Verdict,
    new_id,
    state_from_dict,
    state_to_dict,
)
from .state_machine import ProjectStateMachine
from .judge_clip import (
    FrameListSource,
    FrameSource,
    JudgeConfig,
    NullAudioAnalyzer,
    NullContentAnalyzer,
    VideoFrameSource,
    check_clarity,
    check_duration,
    check_exposure,
    check_near_duplicate,
    check_shake,
    clip_hash,
    judge_clip,
)

__all__ = [
    "Stage", "ShotStatus", "Verdict", "GateType",
    "IntentCard", "Shot", "MaterialCard", "DetectorResult",
    "TechnicalJudgement", "GateEvent", "StateEvent", "ProjectState",
    "ProjectStateMachine",
    "JudgeConfig", "FrameSource", "FrameListSource", "VideoFrameSource",
    "NullContentAnalyzer", "NullAudioAnalyzer",
    "check_duration", "check_clarity", "check_shake", "check_exposure",
    "check_near_duplicate", "clip_hash", "judge_clip",
    "new_id", "state_to_dict", "state_from_dict",
]
