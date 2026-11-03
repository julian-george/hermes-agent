"""Beaker integration: optimize Hermes Agent on Terminal-Bench tasks (TBLite).
Each case runs the real AIAgent in a Python 3.14 worker against the task's own Docker image."""

from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path

from pydantic import BaseModel, Field

from beaker import (
    Case, CaseFile, CaseResult, CaseScore, Check, Integration, RetryableCaseError,
    RolloutRuntime, RunSetup, SetupRuntime, inference_target, repository,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).resolve().with_name("tblite_worker.py")
# Production-like default when the run selects no model; Beaker's provider routing carries it.
DEFAULT_PROVIDER = "openrouter"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-6-luna"
MAX_ITERATIONS = 150
HERMES_PYTHON = "3.14"
# Docker pull, sandbox start and teardown on top of the task's own agent + test budgets.
WORKER_OVERHEAD_SECONDS = 900
STATE_DIR = Path(tempfile.gettempdir()) / "beaker-tblite"


class TaskInput(BaseModel):
    task_name: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    docker_image: str = Field(min_length=1)
    category: str
    difficulty: str
    agent_timeout_sec: int = Field(gt=0)
    test_timeout_sec: int = Field(gt=0)


class TaskExpected(BaseModel):
    test_sh: str = Field(min_length=1)


class Row(BaseModel):
    id: str = Field(min_length=1)
    input: TaskInput
    expected: TaskExpected
    tests_tar_b64: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class Setup(RunSetup[Row]):
    row_model = Row

    async def load_cases(self, row: Row, *, runtime: SetupRuntime) -> AsyncIterator[Case]:
        files_dir = runtime.case_files_dir(row.id)
        tests_tar = files_dir / "tests.tar.gz"
        tests_tar.write_bytes(base64.b64decode(row.tests_tar_b64))
        test_sh = files_dir / "test.sh"
        test_sh.write_text(row.expected.test_sh, encoding="utf-8")
        yield Case(
            id=row.id,
            input=row.input.model_dump(),
            expected=row.expected.model_dump(),
            files=(CaseFile(name="tests.tar.gz", path=tests_tar), CaseFile(name="test.sh", path=test_sh)),
            metadata=row.metadata,
        )


def _locked(name: str):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    handle = open(STATE_DIR / f"{name}.lock", "w")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def _docker_ready() -> bool:
    return subprocess.run(["docker", "info"], capture_output=True, timeout=60, check=False).returncode == 0


def _ensure_docker_daemon() -> None:
    """Start dockerd inside Beaker's VM sandbox once; a no-op where Docker already runs."""
    if _docker_ready():
        return
    with _locked("dockerd"):
        if _docker_ready():
            return
        log = open(STATE_DIR / "dockerd.log", "ab")
        subprocess.Popen(["dockerd"], stdout=log, stderr=log, start_new_session=True)
        for _ in range(120):
            if _docker_ready():
                return
            time.sleep(1)
        raise RuntimeError(f"dockerd did not start; see {STATE_DIR / 'dockerd.log'}")


def _ensure_hermes_python() -> Path:
    """Python 3.14 venv with Hermes's locked dependencies; candidates cannot edit uv.lock."""
    lock_digest = hashlib.sha256((SOURCE_ROOT / "uv.lock").read_bytes()).hexdigest()[:16]
    venv = STATE_DIR / f"hermes-venv-{lock_digest}"
    python = venv / "bin" / "python"
    with _locked("venv"):
        if python.exists():
            return python
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to build the Hermes runtime environment")
        subprocess.run(
            [uv, "sync", "--frozen", "--no-install-project", "--no-default-groups", "--python", HERMES_PYTHON],
            cwd=SOURCE_ROOT, env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)},
            check=True, capture_output=True, timeout=1800,
        )
    return python


def _model_route(runtime: RolloutRuntime) -> tuple[dict, str]:
    if runtime.model:
        target = inference_target(runtime)
        return {"provider": "custom", "base_url": target.base_url, "model": target.model}, target.api_key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is unset; hosted runs get it from Beaker provider routing")
    return {"provider": DEFAULT_PROVIDER, "base_url": DEFAULT_BASE_URL, "model": DEFAULT_MODEL}, api_key


def _replay_trace(runtime: RolloutRuntime, route: dict, events: list[dict]) -> None:
    """Record the worker's model turns and tool results as Beaker spans."""
    tool_names: dict[str, tuple[str, object]] = {}
    for event in events:
        if event["kind"] == "model":
            output = event["output"]
            with runtime.trace.model_call(
                operation="chat", provider=route["provider"], model=route["model"],
                input_messages=event["input_messages"],
            ) as call:
                call.output(output)
            for tool_call in output.get("tool_calls") or []:
                function = tool_call.get("function") or {}
                tool_names[tool_call.get("id")] = (function.get("name", "unknown"), function.get("arguments"))
        else:
            name, arguments = tool_names.get(event["call_id"], ("unknown", None))
            with runtime.trace.tool_call(name, arguments=arguments, call_id=event["call_id"]) as call:
                call.output(event["output"])


async def run_case(*, case_input: dict, runtime: RolloutRuntime) -> CaseResult:
    task = TaskInput.model_validate(case_input)
    route, api_key = _model_route(runtime)
    await asyncio.to_thread(_ensure_docker_daemon)
    python = await asyncio.to_thread(_ensure_hermes_python)
    with tempfile.TemporaryDirectory(prefix=f"tblite-{task.task_name}-") as scratch:
        scratch_dir = Path(scratch)
        home = scratch_dir / "hermes-home"
        home.mkdir()
        job_path, result_path = scratch_dir / "job.json", scratch_dir / "result.json"
        job_path.write_text(json.dumps({
            **task.model_dump(), **route, "max_iterations": MAX_ITERATIONS,
            "tests_tar_path": str(runtime.case_files_dir / "tests.tar.gz"),
            "test_sh_path": str(runtime.case_files_dir / "test.sh"),
        }))
        env = {
            **os.environ,
            "HERMES_HOME": str(home),
            "PYTHONPATH": str(SOURCE_ROOT),
            "TBLITE_MODEL_API_KEY": api_key,
            "TERMINAL_ENV": "docker",
            "TERMINAL_CONTAINER_PERSISTENT": "false",
        }
        worker = await asyncio.create_subprocess_exec(
            str(python), str(WORKER), str(job_path), str(result_path),
            cwd=SOURCE_ROOT, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        budget = task.agent_timeout_sec + task.test_timeout_sec + WORKER_OVERHEAD_SECONDS
        try:
            log, _ = await asyncio.wait_for(worker.communicate(), timeout=budget)
        except BaseException:
            os.killpg(worker.pid, 9)
            await worker.wait()
            raise
        if worker.returncode != 0 or not result_path.exists():
            raise RuntimeError(f"TBLite worker exited {worker.returncode}:\n{log.decode(errors='replace')[-4000:]}")
        result = json.loads(result_path.read_text())
    if result.get("error_kind") == "retryable":
        raise RetryableCaseError(result["error"])
    if result.get("error_kind") == "infra":
        raise RuntimeError(result["error"])
    _replay_trace(runtime, route, result.pop("events"))
    return CaseResult(output=result, output_kind="record")


PYTEST_RESULT = re.compile(r"^(PASSED|FAILED|ERROR) (\S+)", re.MULTILINE)


async def score_case(*, case: Case, result: CaseResult, case_files_dir: Path) -> CaseScore:
    del case, case_files_dir
    output = result.output
    reward = float(output["reward"])
    test_tail = output["test_output_tail"]
    checks = [Check(
        name="hidden test suite", verdict="pass" if reward == 1.0 else "fail",
        message=test_tail if reward < 1.0 else f"reward.txt={output['reward_txt']}",
    )]
    per_test = {name: status for status, name in PYTEST_RESULT.findall(test_tail)}
    checks += [
        Check(name=name, verdict="pass" if status == "PASSED" else "fail", group="pytest")
        for name, status in per_test.items()
    ]
    field_scores = {"passed": reward}
    if per_test:
        field_scores["tests_passed_fraction"] = sum(s == "PASSED" for s in per_test.values()) / len(per_test)
    if output["timed_out"]:
        checks.append(Check(name="agent time limit", verdict="fail", message="Agent hit the task's time limit.", informational=True))
    return CaseScore(objective=reward, field_scores=field_scores, checks=tuple(checks))


integration = Integration(
    targets=repository((
        "agent/prompt_builder.py",
        "agent/context_compressor.py",
        "tools/terminal_tool.py",
        "tools/file_tools.py",
        "skills/software-development",
        "skills/devops",
        "skills/research",
    )),
    run_setup=Setup,
    run_case=run_case,
    score_case=score_case,
)
