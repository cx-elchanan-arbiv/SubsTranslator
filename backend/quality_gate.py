"""
Quality Gate Module - Production-Grade Validation
Ensures subtitle quality before export: no overlaps, proper CPS, timing constraints.
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GateErrorCode(Enum):
    """Error codes for quality gate failures"""

    FAILED_TRANSLATION_JSON = "FAILED_TRANSLATION_JSON"
    FAILED_TRANSLATION_IDS = "FAILED_TRANSLATION_IDS"
    FAILED_TRANSLATION_GATE = "FAILED_TRANSLATION_GATE"
    FAILED_TRANSLATION_TIMEOUT = "FAILED_TRANSLATION_TIMEOUT"
    FAILED_TRANSLATION_OVERLAPS = "FAILED_TRANSLATION_OVERLAPS"
    FAILED_TRANSLATION_CPS = "FAILED_TRANSLATION_CPS"
    FAILED_TRANSLATION_MAX_CUE = "FAILED_TRANSLATION_MAX_CUE"


@dataclass
class GateViolation:
    """Represents a single quality gate violation"""

    code: GateErrorCode
    segment_id: int | None
    message: str
    severity: str  # 'error' or 'warning'


@dataclass
class GateResult:
    """Result of quality gate validation"""

    passed: bool
    violations: list[GateViolation]
    stats: dict[str, any]


def validate_subtitle_quality(
    segments: list[dict],
    max_cps: float = 22.0,
    max_cue_duration: float = 6.0,
    min_gap_ms: float = 50.0,
) -> GateResult:
    """
    Validates subtitle quality before export.

    Args:
        segments: List of subtitle segments with 'start', 'end', 'text' keys
        max_cps: Maximum characters per second
        max_cue_duration: Maximum subtitle duration in seconds
        min_gap_ms: Minimum gap between cues in milliseconds

    Returns:
        GateResult with pass/fail status and violations
    """
    violations = []

    # Statistics
    total_overlaps = 0
    total_cps_violations = 0
    total_long_cues = 0
    cps_values = []

    for i, seg in enumerate(segments):
        seg_id = i + 1
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")

        duration = end - start

        # Check 1: Overlaps with next segment
        if i < len(segments) - 1:
            next_seg = segments[i + 1]
            next_start = next_seg.get("start", 0)
            gap_ms = (next_start - end) * 1000

            if gap_ms < min_gap_ms:
                total_overlaps += 1
                violations.append(
                    GateViolation(
                        code=GateErrorCode.FAILED_TRANSLATION_OVERLAPS,
                        segment_id=seg_id,
                        message=f"Overlap detected: gap={gap_ms:.1f}ms (min={min_gap_ms}ms)",
                        severity="error",
                    )
                )

        # Check 2: CPS (Characters Per Second)
        if duration > 0:
            cps = len(text) / duration
            cps_values.append(cps)

            if cps > max_cps:
                total_cps_violations += 1
                violations.append(
                    GateViolation(
                        code=GateErrorCode.FAILED_TRANSLATION_CPS,
                        segment_id=seg_id,
                        message=f"CPS too high: {cps:.1f} > {max_cps} (text: {len(text)} chars, duration: {duration:.1f}s)",
                        severity="error",
                    )
                )

        # Check 3: Max cue duration
        if duration > max_cue_duration:
            total_long_cues += 1
            violations.append(
                GateViolation(
                    code=GateErrorCode.FAILED_TRANSLATION_MAX_CUE,
                    segment_id=seg_id,
                    message=f"Cue too long: {duration:.1f}s > {max_cue_duration}s",
                    severity="error",
                )
            )

    # Calculate stats
    avg_cps = sum(cps_values) / len(cps_values) if cps_values else 0
    max_cps_found = max(cps_values) if cps_values else 0

    stats = {
        "total_segments": len(segments),
        "overlaps": total_overlaps,
        "cps_violations": total_cps_violations,
        "long_cues": total_long_cues,
        "avg_cps": round(avg_cps, 1),
        "max_cps": round(max_cps_found, 1),
    }

    # Determine if passed
    passed = len([v for v in violations if v.severity == "error"]) == 0

    if passed:
        logger.info(f"✅ Quality gate PASSED: {stats}")
    else:
        logger.error(
            f"❌ Quality gate FAILED: {len(violations)} violations, stats={stats}"
        )

    return GateResult(passed=passed, violations=violations, stats=stats)


def format_gate_error(result: GateResult) -> str:
    """Formats gate errors for user-friendly display"""
    if result.passed:
        return "Translation completed successfully"

    error_msgs = []

    if result.stats["overlaps"] > 0:
        error_msgs.append(f"Found {result.stats['overlaps']} overlapping subtitles")

    if result.stats["cps_violations"] > 0:
        error_msgs.append(f"{result.stats['cps_violations']} subtitles are too long")

    if result.stats["long_cues"] > 0:
        error_msgs.append(
            f"{result.stats['long_cues']} subtitles are displayed too long"
        )

    return "Quality issues: " + ", ".join(error_msgs)
