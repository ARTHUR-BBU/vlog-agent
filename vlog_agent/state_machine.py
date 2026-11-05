"""Project 状态机：阶段推进、回退数据策略、确认门、事件日志、中断恢复。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .schema import (
    GateEvent,
    GateType,
    IntentCard,
    MaterialCard,
    ProjectState,
    Shot,
    ShotStatus,
    Stage,
    StateEvent,
    Verdict,
    new_id,
    now_ts,
    state_from_dict,
    state_to_dict,
)


class ProjectStateMachine:
    def __init__(
        self,
        state: ProjectState | None = None,
        project_id: str | None = None,
    ) -> None:
        self.state = state or ProjectState(project_id=project_id or new_id("proj"))
        self._log("init", {"stage": self.state.stage.name})

    def _log(self, action: str, detail: dict[str, Any]) -> None:
        self.state.events.append(StateEvent(action=action, detail=detail))
        self.state.updated_at = now_ts()

    # ---------- 阶段推进与回退 ----------

    def advance(self, to: Stage, *, reason: str = "") -> None:
        current = self.state.stage
        if to is current:
            return
        if int(to) != int(current) + 1:
            raise ValueError(f"只能逐级推进：{current.name} -> {to.name}")
        if to is Stage.PUBLISH and not self.has_approved_gate(GateType.PRE_PUBLISH):
            raise PermissionError("进入 PUBLISH 前必须通过 pre_publish 确认门")
        self.state.stage = to
        self._log("advance", {"from": current.name, "to": to.name, "reason": reason})

    def rollback(self, to: Stage, *, reason: str = "") -> None:
        current = self.state.stage
        if int(to) >= int(current):
            raise ValueError(f"回退只能往更早阶段：{current.name} -> {to.name}")
        self._apply_rollback_data_policy(to)
        self.state.stage = to
        self._log("rollback", {"from": current.name, "to": to.name, "reason": reason})

    def _apply_rollback_data_policy(self, to: Stage) -> None:
        # 退到 PLAN：改清单，旧清单留版本，等 plan_shotlist 重新生成；已拍素材保留。
        if to is Stage.PLAN and self.state.shotlist:
            self.state.shotlist_versions.append(copy.deepcopy(self.state.shotlist))
            self.state.shotlist = []
        # 退到 SHOOT：补拍。已通过的镜头与素材保留，其余回到待拍，素材不删。
        elif to is Stage.SHOOT:
            for shot in self.state.shotlist:
                if shot.status is not ShotStatus.PASSED:
                    shot.status = ShotStatus.PENDING
        # 其余阶段只改 stage，Artifact 一律保留。

    # ---------- 内容与 Artifact ----------

    def set_intent(self, intent: IntentCard) -> None:
        self.state.intent = intent
        self._log("set_intent", {"topic": intent.topic})

    def set_shotlist(self, shots: list[Shot]) -> None:
        self.state.shotlist = list(shots)
        self._log("set_shotlist", {"count": len(shots)})

    def mark_shot(
        self,
        shot_id: str,
        status: ShotStatus,
        material_id: str | None = None,
    ) -> None:
        shot = self._shot(shot_id)
        shot.status = status
        if material_id and material_id not in shot.material_ids:
            shot.material_ids.append(material_id)
        self._log("mark_shot", {"shot_id": shot_id, "status": status.value})

    def _shot(self, shot_id: str) -> Shot:
        for shot in self.state.shotlist:
            if shot.id == shot_id:
                return shot
        raise KeyError(f"镜头不存在: {shot_id}")

    def attach_material(self, card: MaterialCard) -> None:
        """存素材卡，并按技术判定联动清单状态。"""
        self.state.materials[card.id] = card
        if card.shot_id and card.judgement:
            if card.judgement.passed:
                status = ShotStatus.PASSED
            elif card.judgement.verdict is Verdict.RETAKE:
                status = ShotStatus.RETAKE
            else:
                status = ShotStatus.CAPTURED
            self.mark_shot(card.shot_id, status, card.id)
        self._log("attach_material", {"material_id": card.id, "shot_id": card.shot_id})

    def add_timeline_version(self, timeline: dict[str, Any]) -> None:
        self.state.timeline_versions.append(copy.deepcopy(timeline))
        self._log("add_timeline_version", {"version": len(self.state.timeline_versions)})

    def add_deliverable(self, deliverable: dict[str, Any]) -> None:
        self.state.deliverables.append(copy.deepcopy(deliverable))
        self._log("add_deliverable", {"count": len(self.state.deliverables)})

    def set_review(self, review: dict[str, Any]) -> None:
        self.state.review = copy.deepcopy(review)
        self._log("set_review", {})

    # ---------- 确认门 ----------

    def record_gate(self, gate: GateType, choice: str, *, note: str = "") -> None:
        self.state.gates.append(GateEvent(gate=gate, choice=choice, note=note))
        self._log("gate", {"gate": gate.value, "choice": choice})

    def has_approved_gate(self, gate: GateType) -> bool:
        return any(g.gate is gate and g.choice == "approved" for g in self.state.gates)

    # ---------- 序列化与恢复 ----------

    def to_dict(self) -> dict[str, Any]:
        return state_to_dict(self.state)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProjectStateMachine":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(state=state_from_dict(data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectStateMachine":
        return cls(state=state_from_dict(data))
