"""数据结构：Project 状态、镜头清单、素材卡、判定结果、事件日志。

所有结构保持可 JSON 序列化，支持中断恢复与 Artifact 版本化。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def now_ts() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class Stage(int, Enum):
    INTENT = 0
    PLAN = 1
    SHOOT = 2
    EDIT = 3
    PUBLISH = 4
    REVIEW = 5


class ShotStatus(str, Enum):
    PENDING = "pending"
    CAPTURED = "captured"
    PASSED = "passed"
    RETAKE = "retake"


class Verdict(str, Enum):
    PASS = "pass"
    OPTIONAL_RETAKE = "optional_retake"
    RETAKE = "retake"


class GateType(str, Enum):
    TOPIC_SELECT = "topic_select"
    PRE_PUBLISH = "pre_publish"
    DELETE_PRIVATE_MATERIAL = "delete_private_material"
    PUBLISH_ALL = "publish_all"
    AUTO_PASS = "auto_pass"


@dataclass
class IntentCard:
    topic: str
    audience: str = ""
    mood: str = ""
    primary_platform: str = ""
    target_duration_s: float = 0.0


@dataclass
class Shot:
    id: str
    role: str
    subject: str
    framing: str = ""
    duration_s: float = 3.0
    needs_face: bool = False
    status: ShotStatus = ShotStatus.PENDING
    material_ids: list[str] = field(default_factory=list)


@dataclass
class DetectorResult:
    name: str
    passed: bool
    score: float
    confidence: float
    reason: str = ""


@dataclass
class TechnicalJudgement:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    results: list[DetectorResult] = field(default_factory=list)
    is_hard_fail: bool = False

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.PASS

    def reason_text(self) -> str:
        """现场播报用的一句话。"""
        return "；".join(self.reasons) if self.reasons else "这条能用"


@dataclass
class MaterialCard:
    id: str
    path: str
    duration_s: float = 0.0
    framing: str = ""
    content: str = ""
    human_voice_segments: list[dict[str, Any]] = field(default_factory=list)
    ambient_segments: list[dict[str, Any]] = field(default_factory=list)
    shot_id: str | None = None
    judgement: TechnicalJudgement | None = None


@dataclass
class GateEvent:
    gate: GateType
    choice: str  # approved / rejected / auto
    at: float = field(default_factory=now_ts)
    note: str = ""


@dataclass
class StateEvent:
    action: str
    detail: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=now_ts)


@dataclass
class ProjectState:
    project_id: str
    stage: Stage = Stage.INTENT
    intent: IntentCard | None = None
    shotlist: list[Shot] = field(default_factory=list)
    shotlist_versions: list[list[Shot]] = field(default_factory=list)
    materials: dict[str, MaterialCard] = field(default_factory=dict)
    timeline_versions: list[dict[str, Any]] = field(default_factory=list)
    deliverables: list[dict[str, Any]] = field(default_factory=list)
    review: dict[str, Any] | None = None
    gates: list[GateEvent] = field(default_factory=list)
    events: list[StateEvent] = field(default_factory=list)
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"不可序列化: {type(obj)!r}")


def state_to_dict(state: ProjectState) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(state), default=_json_default))


def _shot_from_dict(data: dict[str, Any]) -> Shot:
    return Shot(**{**data, "status": ShotStatus(data["status"])})


def _judgement_from_dict(data: dict[str, Any] | None) -> TechnicalJudgement | None:
    if data is None:
        return None
    return TechnicalJudgement(
        verdict=Verdict(data["verdict"]),
        reasons=list(data.get("reasons", [])),
        results=[DetectorResult(**r) for r in data.get("results", [])],
        is_hard_fail=bool(data.get("is_hard_fail", False)),
    )


def _material_from_dict(data: dict[str, Any]) -> MaterialCard:
    return MaterialCard(
        id=data["id"],
        path=data["path"],
        duration_s=float(data.get("duration_s", 0.0)),
        framing=data.get("framing", ""),
        content=data.get("content", ""),
        human_voice_segments=list(data.get("human_voice_segments", [])),
        ambient_segments=list(data.get("ambient_segments", [])),
        shot_id=data.get("shot_id"),
        judgement=_judgement_from_dict(data.get("judgement")),
    )


def state_from_dict(data: dict[str, Any]) -> ProjectState:
    return ProjectState(
        project_id=data["project_id"],
        stage=Stage(data["stage"]),
        intent=IntentCard(**data["intent"]) if data.get("intent") else None,
        shotlist=[_shot_from_dict(s) for s in data.get("shotlist", [])],
        shotlist_versions=[
            [_shot_from_dict(s) for s in version]
            for version in data.get("shotlist_versions", [])
        ],
        materials={
            key: _material_from_dict(value)
            for key, value in data.get("materials", {}).items()
        },
        timeline_versions=list(data.get("timeline_versions", [])),
        deliverables=list(data.get("deliverables", [])),
        review=data.get("review"),
        gates=[
            GateEvent(
                gate=GateType(g["gate"]),
                choice=g["choice"],
                at=float(g.get("at", 0.0)),
                note=g.get("note", ""),
            )
            for g in data.get("gates", [])
        ],
        events=[
            StateEvent(
                action=e["action"],
                detail=dict(e.get("detail", {})),
                at=float(e.get("at", 0.0)),
            )
            for e in data.get("events", [])
        ],
        created_at=float(data.get("created_at", 0.0)),
        updated_at=float(data.get("updated_at", 0.0)),
    )
