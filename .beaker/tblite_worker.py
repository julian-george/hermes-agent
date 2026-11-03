"""Run one TBLite task through the real AIAgent, then grade it in the same container.
Usage (fresh Python 3.14 process per case): tblite_worker.py <job.json> <result.json>"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path

INTERRUPT_GRACE_SECONDS = 90
TAIL_BYTES = 4096
# Hermes FailoverReason values that mean the provider, not the agent, failed.
RETRYABLE_FAILURES = {"rate_limit", "upstream_rate_limit", "overloaded", "server_error", "timeout", "unknown"}
INFRA_FAILURES = {
    "auth", "auth_permanent", "billing", "model_not_found", "model_entitlement",
    "ssl_cert_verification", "provider_policy_blocked", "upstream_blocked",
}
AUXILIARY_TASKS = (
    "vision", "compression", "skills_hub", "approval", "review", "mcp", "title_generation",
    "memory_query_rewrite", "tts_audio_tags", "voice_chat", "triage_specifier", "kanban_decomposer",
    "profile_describer", "goal_judge", "curator", "monitor", "background_review",
)


def tail(text: str) -> str:
    return text[-TAIL_BYTES:]


def write_hermes_config(home: Path, *, provider: str, base_url: str, model: str) -> None:
    """Pin the main and every auxiliary model to one endpoint, with no fallback chain."""
    endpoint = {"provider": provider, "base_url": base_url, "api_key": "${TBLITE_MODEL_API_KEY}", "model": model}
    config = {
        "model": {"provider": provider, "base_url": base_url, "api_key": "${TBLITE_MODEL_API_KEY}", "default": model},
        "fallback_providers": [],
        "auxiliary": {task: dict(endpoint) for task in AUXILIARY_TASKS},
    }
    # JSON is valid YAML, so no YAML library is needed here.
    (home / "config.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_agent(job: dict, task_id: str) -> dict:
    from run_agent import AIAgent

    agent = AIAgent(
        model=job["model"], base_url=job["base_url"], api_key=os.environ["TBLITE_MODEL_API_KEY"],
        provider=job["provider"], api_mode="chat_completions", fallback_model=None,
        enabled_toolsets=["terminal", "file"], max_iterations=job["max_iterations"],
        run_budget_seconds=job["agent_timeout_sec"],
        skip_context_files=True, skip_memory=True, skip_background_review=True,
        quiet_mode=True, platform="batch",
    )
    outcome: dict = {}

    def converse() -> None:
        try:
            outcome["result"] = agent.run_conversation(job["instruction"], task_id=task_id)
        except BaseException as exc:  # handed to the main thread, which classifies it
            outcome["exception"] = exc

    thread = threading.Thread(target=converse, name="tblite-agent", daemon=True)
    started = time.monotonic()
    thread.start()
    thread.join(job["agent_timeout_sec"])
    timed_out = thread.is_alive()
    if timed_out:
        agent.interrupt("Time limit reached.", hard_cancel=True)
        thread.join(INTERRUPT_GRACE_SECONDS)
    elapsed = time.monotonic() - started
    try:
        agent.close()
    except Exception as exc:  # the container is graded and removed by the caller regardless
        print(f"agent.close failed: {exc!r}", file=sys.stderr)
    if "exception" in outcome:
        raise outcome["exception"]
    result = outcome.get("result") or {}
    return {"result": result, "timed_out": timed_out, "elapsed_sec": elapsed}


def container_id_for(task_id: str) -> tuple[str, str]:
    from tools.terminal_tool_lifecycle import ensure_task_env, get_active_env

    env = get_active_env(task_id) or ensure_task_env(task_id)
    if env is None or not getattr(env, "_container_id", None):
        raise RuntimeError(f"No Docker sandbox exists for task {task_id}")
    return env._docker_exe, env._container_id


def grade(job: dict, docker: str, container: str) -> dict:
    """Copy the hidden tests in only now, run test.sh from /app and read the reward."""
    with tempfile.TemporaryDirectory(prefix="tblite-tests-") as staging:
        tests_dir = Path(staging) / "tests"
        tests_dir.mkdir()
        with tarfile.open(job["tests_tar_path"], "r:gz") as archive:
            archive.extractall(tests_dir, filter="data")
        (tests_dir / "test.sh").write_text(Path(job["test_sh_path"]).read_text(encoding="utf-8"), encoding="utf-8")
        subprocess.run([docker, "exec", container, "mkdir", "-p", "/tests", "/logs/verifier"], check=True, timeout=60)
        subprocess.run([docker, "cp", f"{tests_dir}/.", f"{container}:/tests/"], check=True, timeout=300)
    try:
        run = subprocess.run(
            [docker, "exec", "-w", "/app", container, "bash", "/tests/test.sh"],
            capture_output=True, text=True, errors="replace", timeout=job["test_timeout_sec"], check=False,
        )
        test_output, exit_code = run.stderr + run.stdout, run.returncode
    except subprocess.TimeoutExpired as exc:
        test_output = (exc.stdout or b"").decode(errors="replace") + (exc.stderr or b"").decode(errors="replace")
        test_output += f"\n[test.sh exceeded {job['test_timeout_sec']}s]"
        exit_code = -1
    reward_file = subprocess.run(
        [docker, "exec", container, "cat", "/logs/verifier/reward.txt"], capture_output=True, text=True, timeout=60, check=False,
    )
    reward_text = reward_file.stdout.strip() if reward_file.returncode == 0 else ""
    if reward_text:
        reward = 1.0 if reward_text == "1" else 0.0
    else:
        reward = 1.0 if exit_code == 0 else 0.0
    return {"reward": reward, "test_exit_code": exit_code, "reward_txt": reward_text or None, "test_output": test_output}


def tool_events(messages: list[dict]) -> list[dict]:
    """Model turns and tool results, in order, for the caller to replay into the Beaker trace."""
    events: list[dict] = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            events.append({"kind": "model", "input_messages": messages[:index], "output": message})
        elif message.get("role") == "tool":
            events.append({"kind": "tool", "call_id": message.get("tool_call_id"), "output": message.get("content")})
    return events


def main(job_path: str, result_path: str) -> None:
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    task_id = f"tblite-{job['task_name']}-{uuid.uuid4().hex[:8]}"
    home = Path(os.environ["HERMES_HOME"])
    write_hermes_config(home, provider=job["provider"], base_url=job["base_url"], model=job["model"])

    from tools.skills_sync import sync_skills
    from tools.terminal_tool import register_task_env_overrides
    from tools.terminal_tool_lifecycle import cleanup_vm

    sync_skills(quiet=True)
    image = job["docker_image"]
    pull = subprocess.run(["docker", "pull", image], capture_output=True, text=True, timeout=1200, check=False)
    if pull.returncode != 0:
        Path(result_path).write_text(json.dumps({"error_kind": "retryable", "error": f"docker pull {image} failed: {tail(pull.stderr)}"}))
        return
    register_task_env_overrides(task_id, {"env_type": "docker", "docker_image": image, "cwd": "/app"})
    try:
        run = run_agent(job, task_id)
        result = run["result"]
        failure = result.get("failure_reason")
        if result.get("failed") and failure in RETRYABLE_FAILURES | INFRA_FAILURES:
            kind = "retryable" if failure in RETRYABLE_FAILURES else "infra"
            Path(result_path).write_text(json.dumps({"error_kind": kind, "error": f"{failure}: {tail(str(result.get('error')))}"}))
            return
        docker, container = container_id_for(task_id)
        graded = grade(job, docker, container)
    finally:
        cleanup_vm(task_id, force_remove=True)
    messages = result.get("messages") or []
    Path(result_path).write_text(json.dumps({
        "reward": graded["reward"],
        "finished": bool(result.get("completed")) and not run["timed_out"],
        "timed_out": run["timed_out"],
        "agent_error": result.get("error") if result.get("failed") else None,
        "turns": sum(1 for message in messages if message.get("role") == "assistant"),
        "api_calls": result.get("api_calls", 0),
        "elapsed_sec": round(run["elapsed_sec"], 1),
        "test_exit_code": graded["test_exit_code"],
        "reward_txt": graded["reward_txt"],
        "test_output_tail": tail(graded["test_output"]),
        "final_response_tail": tail(str(result.get("final_response") or "")),
        "events": tool_events(messages),
    }, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:3])
