"""judge_clip 技术判定：确定性检测项 + 置信度分级汇合。

设计要点（见 docs/product-tech-skeleton.md 2.5）：
- 技术质量用确定性算法，不用模型，可离线单测；
- 判定结果必须带可解释原因，现场要直接播报给用户；
- 误判代价不对称：把好片判成糊、让人白重拍，比漏过一条差片更伤体验，
  所以边界附近只给「可选重拍」，不强制。
- 内容理解与音频分层留扩展点，第一版不接模型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, Sequence

import cv2
import numpy as np

from .schema import (
    DetectorResult,
    MaterialCard,
    TechnicalJudgement,
    Verdict,
    new_id,
)


@dataclass
class JudgeConfig:
    min_duration_s: float = 1.5
    clarity_threshold: float = 100.0
    shake_threshold: float = 6.0
    dark_threshold: float = 40.0
    bright_threshold: float = 215.0
    duplicate_hamming: int = 5
    hard_fail_confidence: float = 0.8
    sample_frames: int = 24
    low_confidence_band: float = 0.25


class FrameSource(Protocol):
    duration_s: float
    fps: float

    def frames(self) -> Iterator[np.ndarray]: ...


@dataclass
class FrameListSource:
    """内存帧源，用于离线单测与上层自行抽帧。"""
    frames_list: list[np.ndarray]
    fps: float = 30.0
    duration_s: float = 0.0

    def frames(self) -> Iterator[np.ndarray]:
        yield from self.frames_list


class VideoFrameSource:
    """从视频文件均匀抽帧。抽帧数受 JudgeConfig.sample_frames 控制。"""
    def __init__(self, path: str, sample_frames: int = 24) -> None:
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"无法打开素材: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_s = total / self.fps if total and self.fps else 0.0
        count = min(sample_frames, max(total, 1))
        indices = np.linspace(0, max(total - 1, 0), count, dtype=int) if total else []
        self._frames: list[np.ndarray] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                self._frames.append(frame)
        cap.release()
        if not self._frames:
            raise ValueError(f"素材没有可读帧: {path}")

    def frames(self) -> Iterator[np.ndarray]:
        yield from self._frames


class ContentAnalyzer(Protocol):
    """后续接多模态模型：返回 (画面内容, 景别)。"""
    def analyze(self, frames: Sequence[np.ndarray]) -> tuple[str, str]: ...


class AudioAnalyzer(Protocol):
    """后续接音频分析：返回 (人声段, 环境音可用段)。"""
    def analyze(self, path: str) -> tuple[list[dict], list[dict]]: ...


class NullContentAnalyzer:
    def analyze(self, frames: Sequence[np.ndarray]) -> tuple[str, str]:
        return "", ""


class NullAudioAnalyzer:
    def analyze(self, path: str) -> tuple[list[dict], list[dict]]:
        return [], []


@dataclass
class _Borderline:
    """把「超出阈值多少」映射成置信度：边界附近低置信度，明显越界高置信度。"""
    score: float
    threshold: float
    band: float
    higher_is_worse: bool

    def result(self, name: str, reason: str) -> DetectorResult:
        if self.higher_is_worse:
            passed = self.score <= self.threshold
            excess = (self.score - self.threshold) / max(self.threshold, 1e-6)
        else:
            passed = self.score >= self.threshold
            excess = (self.threshold - self.score) / max(self.threshold, 1e-6)
        if passed:
            return DetectorResult(name=name, passed=True, score=float(self.score),
                                  confidence=1.0)
        confidence = min(1.0, max(0.0, excess / self.band))
        return DetectorResult(name=name, passed=False, score=float(self.score),
                              confidence=confidence, reason=reason)


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def check_duration(source: FrameSource, config: JudgeConfig) -> DetectorResult:
    score = float(source.duration_s)
    if score >= config.min_duration_s:
        return DetectorResult("duration", True, score, 1.0)
    return DetectorResult(
        "duration", False, score, 1.0,
        reason=f"时长不足 {config.min_duration_s}s，重拍一条",
    )


def check_clarity(
    frames: Sequence[np.ndarray], config: JudgeConfig
) -> DetectorResult:
    if not frames:
        return DetectorResult("clarity", False, 0.0, 1.0, reason="没有可读画面")
    scores = [
        float(cv2.Laplacian(_gray(f), cv2.CV_64F).var()) for f in frames
    ]
    score = float(np.median(scores))
    return _Borderline(score, config.clarity_threshold, config.low_confidence_band,
                       higher_is_worse=False).result("clarity", "画面糊了，重拍一条")


def check_shake(
    frames: Sequence[np.ndarray], config: JudgeConfig
) -> DetectorResult:
    if len(frames) < 2:
        return DetectorResult("shake", True, 0.0, 1.0)
    small = [
        cv2.resize(_gray(f), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        for f in frames
    ]
    diffs = [float(np.mean(np.abs(small[i] - small[i - 1])))
             for i in range(1, len(small))]
    score = float(np.median(diffs))
    return _Borderline(score, config.shake_threshold, config.low_confidence_band,
                       higher_is_worse=True).result("shake", "画面太抖，重拍一条")


def check_exposure(
    frames: Sequence[np.ndarray], config: JudgeConfig
) -> DetectorResult:
    if not frames:
        return DetectorResult("exposure", False, 0.0, 1.0, reason="没有可读画面")
    means = [float(np.mean(_gray(f))) for f in frames]
    score = float(np.median(means))
    if config.dark_threshold <= score <= config.bright_threshold:
        return DetectorResult("exposure", True, score, 1.0)
    if score < config.dark_threshold:
        reason, threshold = "太暗了，补点光重拍", config.dark_threshold
        border = _Borderline(score, threshold, config.low_confidence_band, False)
    else:
        reason, threshold = "太亮了，换个角度重拍", config.bright_threshold
        border = _Borderline(score, threshold, config.low_confidence_band, True)
    return border.result("exposure", reason)


def _dhash(gray: np.ndarray, size: int = 8) -> np.ndarray:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def check_near_duplicate(
    frames: Sequence[np.ndarray],
    known_hashes: Sequence[np.ndarray],
    config: JudgeConfig,
) -> DetectorResult:
    if not frames or not known_hashes:
        return DetectorResult("near_duplicate", True, 0.0, 1.0)
    current = _dhash(_gray(frames[0]))
    distance = min(_hamming(current, h) for h in known_hashes)
    if distance > config.duplicate_hamming:
        return DetectorResult("near_duplicate", True, float(distance), 1.0)
    return DetectorResult(
        "near_duplicate", False, float(distance), 0.9,
        reason="和已有素材重复，换一个角度拍",
    )


def clip_hash(source: FrameSource) -> np.ndarray | None:
    frames = list(source.frames())
    return _dhash(_gray(frames[0])) if frames else None


def judge_clip(
    source: FrameSource,
    *,
    config: JudgeConfig | None = None,
    known_hashes: Sequence[np.ndarray] | None = None,
    content_analyzer: ContentAnalyzer | None = None,
    audio_analyzer: AudioAnalyzer | None = None,
    material_id: str | None = None,
    path: str = "",
    shot_id: str | None = None,
) -> MaterialCard:
    """技术判定汇合，产出带可解释原因的素材卡。"""
    cfg = config or JudgeConfig()
    frames = list(source.frames())
    results = [
        check_duration(source, cfg),
        check_clarity(frames, cfg),
        check_shake(frames, cfg),
        check_exposure(frames, cfg),
        check_near_duplicate(frames, known_hashes or [], cfg),
    ]

    hard = [r for r in results
            if not r.passed and r.confidence >= cfg.hard_fail_confidence]
    soft = [r for r in results
            if not r.passed and r.confidence < cfg.hard_fail_confidence]

    if hard:
        verdict, is_hard = Verdict.RETAKE, True
    elif soft:
        verdict, is_hard = Verdict.OPTIONAL_RETAKE, False
    else:
        verdict, is_hard = Verdict.PASS, False

    judgement = TechnicalJudgement(
        verdict=verdict,
        reasons=[r.reason for r in (hard + soft)],
        results=results,
        is_hard_fail=is_hard,
    )

    content, framing = (content_analyzer or NullContentAnalyzer()).analyze(frames)
    human, ambient = (audio_analyzer or NullAudioAnalyzer()).analyze(path)

    return MaterialCard(
        id=material_id or new_id("mat"),
        path=path,
        duration_s=float(source.duration_s),
        framing=framing,
        content=content,
        human_voice_segments=human,
        ambient_segments=ambient,
        shot_id=shot_id,
        judgement=judgement,
    )
