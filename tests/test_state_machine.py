import pytest

from vlog_agent.schema import (
    GateType,
    IntentCard,
    MaterialCard,
    Shot,
    ShotStatus,
    Stage,
    TechnicalJudgement,
    Verdict,
    state_from_dict,
    state_to_dict,
)
from vlog_agent.state_machine import ProjectStateMachine


def _shots(n: int) -> list[Shot]:
    return [
        Shot(id=f"s{i}", role="过程", subject=f"镜头{i}", framing="中景")
        for i in range(n)
    ]


def _to_edit() -> ProjectStateMachine:
    sm = ProjectStateMachine(project_id="p1")
    sm.set_intent(IntentCard(topic="周末去乡下"))
    sm.advance(Stage.PLAN)
    sm.set_shotlist(_shots(3))
    sm.advance(Stage.SHOOT)
    sm.advance(Stage.EDIT)
    return sm


def test_forward_and_illegal_jump():
    sm = _to_edit()
    assert sm.state.stage is Stage.EDIT
    with pytest.raises(ValueError):
        sm.advance(Stage.REVIEW)
    with pytest.raises(ValueError):
        sm.rollback(Stage.PUBLISH)


def test_publish_requires_gate():
    sm = _to_edit()
    with pytest.raises(PermissionError):
        sm.advance(Stage.PUBLISH)
    sm.record_gate(GateType.PRE_PUBLISH, "approved")
    sm.advance(Stage.PUBLISH)
    assert sm.state.stage is Stage.PUBLISH
    assert sm.has_approved_gate(GateType.PRE_PUBLISH)


def test_rollback_to_plan_keeps_shotlist_version():
    sm = _to_edit()
    original_ids = [s.id for s in sm.state.shotlist]
    sm.rollback(Stage.PLAN, reason="主题不对")
    assert len(sm.state.shotlist_versions) == 1
    assert [s.id for s in sm.state.shotlist_versions[0]] == original_ids
    assert sm.state.shotlist == []
    assert sm.state.stage is Stage.PLAN


def test_rollback_to_shoot_keeps_passed_and_materials():
    sm = _to_edit()
    sm.mark_shot("s0", ShotStatus.PASSED, "m0")
    sm.mark_shot("s1", ShotStatus.CAPTURED)
    sm.attach_material(
        MaterialCard(
            id="m0", path="a.mp4", duration_s=3.0, shot_id="s0",
            judgement=TechnicalJudgement(verdict=Verdict.PASS),
        )
    )
    sm.rollback(Stage.SHOOT, reason="剪辑时发现缺镜头")
    by_id = {s.id: s for s in sm.state.shotlist}
    assert by_id["s0"].status is ShotStatus.PASSED
    assert by_id["s1"].status is ShotStatus.PENDING
    assert by_id["s2"].status is ShotStatus.PENDING
    assert "m0" in sm.state.materials
    assert by_id["s0"].material_ids == ["m0"]


def test_gate_and_events_are_append_only():
    sm = ProjectStateMachine(project_id="p2")
    before = len(sm.state.events)
    sm.record_gate(GateType.TOPIC_SELECT, "approved", note="选了 A")
    sm.record_gate(GateType.AUTO_PASS, "auto")
    assert len(sm.state.events) == before + 2
    assert len(sm.state.gates) == 2
    assert sm.state.events[-1].detail["choice"] == "auto"


def test_json_roundtrip_resume(tmp_path):
    sm = _to_edit()
    sm.mark_shot("s0", ShotStatus.PASSED, "m0")
    sm.attach_material(
        MaterialCard(
            id="m0", path="a.mp4", duration_s=3.0, shot_id="s0",
            judgement=TechnicalJudgement(
                verdict=Verdict.RETAKE,
                reasons=["画面糊了，重拍一条"],
                is_hard_fail=True,
            ),
        )
    )
    sm.add_timeline_version({"order": ["m0"], "length_s": 3.0})
    sm.add_deliverable({"platform": "douyin", "title": "周末"})
    sm.record_gate(GateType.TOPIC_SELECT, "approved")

    restored = ProjectStateMachine.from_dict(state_to_dict(sm.state))
    assert restored.state.project_id == sm.state.project_id
    assert restored.state.stage is Stage.EDIT
    assert restored.state.materials["m0"].judgement.verdict is Verdict.RETAKE
    assert restored.state.materials["m0"].judgement.reason_text() == "画面糊了，重拍一条"
    # attach_material 按判定结果联动了清单状态，RETAKE 才是这里的预期值
    assert restored.state.shotlist[0].status is ShotStatus.RETAKE
    assert restored.state.shotlist[0].material_ids == ["m0"]
    assert restored.state.timeline_versions == [{"order": ["m0"], "length_s": 3.0}]
    assert restored.state.deliverables[0]["platform"] == "douyin"
    assert len(restored.state.gates) == 1

    path = tmp_path / "project.json"
    restored.save(path)
    loaded = ProjectStateMachine.load(path)
    assert loaded.state.stage is Stage.EDIT
    assert loaded.state.materials["m0"].judgement.is_hard_fail is True


def test_state_from_dict_is_independent():
    sm = _to_edit()
    data = state_to_dict(sm.state)
    rebuilt = state_from_dict(data)
    rebuilt.shotlist[0].status = ShotStatus.RETAKE
    assert sm.state.shotlist[0].status is ShotStatus.PENDING
