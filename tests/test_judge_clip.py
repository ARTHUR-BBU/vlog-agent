import cv2
import numpy as np

from vlog_agent.judge_clip import (
    FrameListSource,
    JudgeConfig,
    clip_hash,
    judge_clip,
)
from vlog_agent.schema import Verdict


def _pattern(seed: int = 1, shift: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(0, 240, 20):
        for x in range(0, 320, 20):
            img[y:y + 20, x:x + 20] = int(rng.integers(20, 235))
    if shift:
        matrix = np.float32([[1, 0, shift], [0, 1, shift // 2]])
        img = cv2.warpAffine(img, matrix, (320, 240))
    return img


def _source(frames, duration_s: float = 3.0) -> FrameListSource:
    return FrameListSource(frames_list=list(frames), fps=30.0, duration_s=duration_s)


def test_clear_clip_passes():
    card = judge_clip(_source([_pattern(1) for _ in range(8)]))
    assert card.judgement.verdict is Verdict.PASS
    assert card.judgement.is_hard_fail is False
    assert card.judgement.reasons == []
    assert card.judgement.reason_text() == "这条能用"


def test_blurred_clip_is_hard_fail_with_reason():
    frames = [cv2.GaussianBlur(_pattern(1), (31, 31), 0) for _ in range(8)]
    card = judge_clip(_source(frames))
    assert card.judgement.verdict is Verdict.RETAKE
    assert card.judgement.is_hard_fail is True
    assert any("糊" in r for r in card.judgement.reasons)


def test_short_clip_is_hard_fail():
    card = judge_clip(_source([_pattern(1)], duration_s=0.8))
    assert card.judgement.verdict is Verdict.RETAKE
    assert any("时长" in r for r in card.judgement.reasons)


def test_shaky_clip_is_detected():
    frames = [_pattern(1, shift=i * 20) for i in range(8)]
    card = judge_clip(_source(frames))
    assert card.judgement.verdict is Verdict.RETAKE
    assert any("抖" in r for r in card.judgement.reasons)


def test_dark_clip_is_detected():
    frames = [(_pattern(1) * 0.15).astype(np.uint8) for _ in range(8)]
    card = judge_clip(_source(frames))
    assert card.judgement.verdict is Verdict.RETAKE
    assert any("暗" in r for r in card.judgement.reasons)


def test_borderline_is_optional_retake_not_forced():
    """误判代价不对称：边界附近只给可选重拍，不强制用户重拍。"""
    frames = [_pattern(1) for _ in range(8)]
    from vlog_agent.judge_clip import check_clarity
    score = check_clarity(frames, JudgeConfig()).score
    cfg = JudgeConfig(clarity_threshold=score * 1.02)
    card = judge_clip(_source(frames), config=cfg)
    assert card.judgement.verdict is Verdict.OPTIONAL_RETAKE
    assert card.judgement.is_hard_fail is False
    assert card.judgement.reasons  # 仍然给得出原因


def test_near_duplicate_detection():
    frames = [_pattern(1) for _ in range(8)]
    known = [clip_hash(_source(frames))]
    card = judge_clip(_source(frames), known_hashes=known)
    assert card.judgement.verdict is Verdict.RETAKE
    assert any("重复" in r for r in card.judgement.reasons)


def test_results_are_explainable():
    card = judge_clip(_source([_pattern(1)], duration_s=0.8))
    assert len(card.judgement.results) == 5
    assert all(r.name for r in card.judgement.results)
    assert card.judgement.reason_text()


def test_content_analyzer_extension_point():
    class FakeAnalyzer:
        def analyze(self, frames):
            return "一杯咖啡被端起来", "特写"

    card = judge_clip(
        _source([_pattern(1) for _ in range(8)]),
        content_analyzer=FakeAnalyzer(),
        shot_id="s3",
    )
    assert card.content == "一杯咖啡被端起来"
    assert card.framing == "特写"
    assert card.shot_id == "s3"
