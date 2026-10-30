# Copyright 2025 - Frontier-CS reward for VERL RL training
"""Frontier-CS judge reward: submit C++ code to judge API, return score.

Returns a dict so that judge-side failures are visible in the logged metrics instead
of silently becoming 0 rewards:

    score         reward used for training / validation (0-100; 0 on any failure)
    acc           same as score (keeps verl's val-core metric name stable)
    judge_status  1 judged, 0 no code block in the response, -1 timed out waiting
                  for the judge, -2 client/HTTP error after retries, -3 judge error
                  (e.g. compile failed; a legitimate 0)

Timeouts (env): FRONTIERCS_MAX_WAIT seconds to wait for a verdict (default 1800 —
a Frontier-CS validation sends ~860 submissions at once and single jobs can run for
many minutes; the original 300 s turned most of a validation into zeros, while single
pathological jobs have run for 75 min, so the wait is capped and reported as -1).
"""

import os
import re
import time
from typing import Any, Optional

import requests

DEFAULT_JUDGE_URL = "http://localhost:8082"
POLL_INTERVAL = 2.0
MAX_WAIT = float(os.environ.get("FRONTIERCS_MAX_WAIT", "1800"))
HTTP_RETRIES = int(os.environ.get("FRONTIERCS_HTTP_RETRIES", "20"))
SUBMIT_TIMEOUT = 120
POLL_TIMEOUT = 60
HARDTEST_PREFIXES = ("hardtest_smp_", "hardtest_orig_")
HARDTEST_FULL_SCORE = 100.0

STATUS_JUDGED, STATUS_NO_CODE, STATUS_TIMEOUT, STATUS_CLIENT_ERROR, STATUS_JUDGE_ERROR = 1, 0, -1, -2, -3


def strip_think(response: str) -> str:
    """Remove everything through the last closing think tag."""
    if not response:
        return response
    _, sep, suffix = response.rpartition("</think>")
    return suffix if sep else response


def extract_cpp(response_text: str) -> str:
    """Extract C++ code from model response (markdown or raw), ignoring <think> blocks."""
    if not response_text:
        return ""

    response_text = strip_think(response_text)
    if not response_text:
        return ""

    code = response_text.strip()

    # Try to extract from ```cpp blocks
    cpp_pattern = r'```(?:cpp|c\+\+)?\s*\n(.*?)```'
    matches = re.findall(cpp_pattern, code, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # Fallback: strip markdown if present
    if code.startswith("```cpp"):
        code = code[6:].strip()
    elif code.startswith("```c++"):
        code = code[6:].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    return code


def is_hardtest_problem(ground_truth: Any) -> bool:
    """Whether this problem id is one of the packaged hardtest variants."""
    return str(ground_truth).startswith(HARDTEST_PREFIXES)


def normalize_frontiercs_score(raw_score: float, ground_truth: Any) -> float:
    """Map hardtest judge scores to binary reward while preserving default behavior otherwise."""
    if is_hardtest_problem(ground_truth):
        return 100.0 if raw_score >= HARDTEST_FULL_SCORE else 0.0
    return raw_score


def _result(score: float, status: int) -> dict:
    return {"score": float(score), "acc": float(score), "judge_status": float(status)}


def _with_retries(fn, what: str):
    """Call fn() with retries and backoff; raises the last exception after HTTP_RETRIES failures."""
    delay = 1.0
    for attempt in range(HTTP_RETRIES):
        try:
            return fn()
        except Exception:
            if attempt == HTTP_RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    judge_url: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Submit C++ solution to Frontier-CS judge and return the score (see module docstring).

    Hardtest package ids (hardtest_smp_* / hardtest_orig_*) are treated as
    strict binary reward: only full score counts as 100, partial score counts
    as 0.0. All other Frontier-CS problems keep the original 0-100 score.

    Args:
        data_source: Must be "frontiercs"
        solution_str: Model's raw response (may contain ```cpp ... ```)
        ground_truth: problem_id (str)
        judge_url: Judge API base URL (default: http://localhost:8082)

    Returns:
        dict: {"score": reward (0 on any failure), "acc": same, "judge_status": see module docstring}
    """
    if data_source != "frontiercs":
        raise ValueError(f"data_source must be 'frontiercs', got {data_source}")

    url = (judge_url or extra_info.get("judge_url") if extra_info else None) or DEFAULT_JUDGE_URL
    url = url.rstrip("/")
    problem_id = str(ground_truth)
    code = extract_cpp(solution_str)

    if not code:
        return _result(0.0, STATUS_NO_CODE)

    def submit():
        r = requests.post(
            f"{url}/submit",
            files={"code": ("sol.cpp", code)},
            data={"pid": problem_id, "lang": "cpp"},
            timeout=SUBMIT_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("sid")

    def poll(sid):
        r2 = requests.get(f"{url}/result/{sid}", timeout=POLL_TIMEOUT)
        if r2.status_code == 404:
            return None  # not ready yet
        r2.raise_for_status()
        return r2.json()

    try:
        sid = _with_retries(submit, "submit")
        if not sid:
            return _result(0.0, STATUS_CLIENT_ERROR)

        start = time.time()
        while time.time() - start < MAX_WAIT:
            res = _with_retries(lambda: poll(sid), "poll")
            if res is not None:
                status = res.get("status")
                if status == "done":
                    raw_score = float(res.get("score", 0))
                    return _result(normalize_frontiercs_score(raw_score, problem_id), STATUS_JUDGED)
                if status == "error":
                    return _result(0.0, STATUS_JUDGE_ERROR)
            time.sleep(POLL_INTERVAL)
        return _result(0.0, STATUS_TIMEOUT)
    except Exception:
        return _result(0.0, STATUS_CLIENT_ERROR)
