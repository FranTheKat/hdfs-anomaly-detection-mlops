"""Validación estática y de comportamiento de los workflows de GitHub Actions.
No ejecuta GitHub Actions: comprueba su estructura y ejecuta localmente el script del gate."""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest
import yaml
from conftest import is_repository_root

from hdfs_anomaly.config import REPO_ROOT

WF = REPO_ROOT / ".github/workflows"


def load(name):
    data = yaml.safe_load((WF / name).read_text(encoding="utf-8"))
    data["on"] = data.pop(True, data.get("on"))  # PyYAML interpreta la clave «on» como True
    return data


def steps(job):
    return job.get("steps", [])


def runs(job):
    return "\n".join(s.get("run", "") for s in steps(job))


CI, CD = load("ci.yml"), load("cd.yml")


def test_ci_triggers_and_minimal_permissions():
    assert set(CI["on"]) == {"push", "pull_request"}
    assert CI["permissions"] == {"contents": "read"}
    assert all("permissions" not in j or j["permissions"] == {"contents": "read"} for j in CI["jobs"].values())
    text = (WF / "ci.yml").read_text(encoding="utf-8")
    assert "pull_request_target" not in text.replace("nunca pull_request_target", "")
    assert "workflow_run" not in text and "secrets." not in text


def test_ci_runs_required_checks_without_dataset_or_test_evaluation():
    test, docker = CI["jobs"]["test"], CI["jobs"]["docker"]
    for job in (test, docker):
        py = [s for s in steps(job) if s.get("uses", "").startswith("actions/setup-python")]
        assert py and py[0]["with"]["python-version"] == "3.14.7"
        assert all(s["with"].get("persist-credentials") is False
                   for s in steps(job) if s.get("uses", "").startswith("actions/checkout"))
        assert "requirements-dev.lock.txt" in runs(job)
    t = runs(test)
    assert "ruff check ." in t and "pytest" in t and "tests/test_api.py" in t and "release decide" in t
    d = runs(docker)
    assert "docker build -t hdfs-anomaly:demo-local" in d and "APP_MODE=production" in d
    assert any(s.get("env", {}).get("RUN_DOCKER_TESTS") == "1" for s in steps(docker))
    both = t + d
    assert "evaluate-test" not in both and "train select" not in both and "HDFS_V1_DIR" not in both
    assert "docker push" not in both and "docker login" not in both


def test_cd_only_publishes_after_verified_gate():
    assert set(CD["on"]) == {"push"} and CD["on"]["push"]["branches"] == ["main"]
    assert "workflow_dispatch" not in (WF / "cd.yml").read_text(encoding="utf-8")  # sin disparo manual con entradas
    assert CD["permissions"] == {}
    gate, publish = CD["jobs"]["gate"], CD["jobs"]["publish"]
    assert gate["permissions"] == {"contents": "read"}
    assert publish["permissions"] == {"contents": "read", "packages": "write"}
    assert publish["needs"] == "gate" and publish["if"] == "needs.gate.outputs.approved == 'true'"
    names = [s.get("name", "") for s in steps(publish)]
    runs_ = [s.get("run", "") for s in steps(publish)]
    verify = next(i for i, r in enumerate(runs_) if "release decide" in r)
    push = next(i for i, r in enumerate(runs_) if "docker push" in r)
    assert verify < push, names  # re-verificación antes de publicar
    assert "docker push" not in runs(gate) and "packages" not in str(gate.get("permissions"))
    text = (WF / "cd.yml").read_text(encoding="utf-8")
    assert not re.search(r"inputs\.|FORCE|SKIP_GATE|OVERRIDE", text)


def _gate_script():
    return next(s["run"] for s in steps(CD["jobs"]["gate"]) if s.get("id") == "decide")


def _bash():
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")  # en Windows, el bash de Git for Windows (no el de WSL)
    cand = os.path.join(os.path.dirname(os.path.dirname(git)), "bin", "bash.exe") if git else ""
    return cand if os.path.exists(cand) else None


def test_ci_accepts_both_valid_decisions_and_only_fails_on_integrity():
    step = next(s for s in steps(CI["jobs"]["test"]) if s.get("id") == "release")["run"]
    assert "0) echo \"approved=true\"" in step and "3) echo \"approved=false\"" in step and "exit 1" in step
    assert "gate']=='failed'" not in step  # la CI no supone que el modelo esté rechazado
    assert CI["jobs"]["test"]["outputs"]["approved"] == "${{ steps.release.outputs.approved }}"
    assert "tests/test_release.py" in runs(CI["jobs"]["test"])  # ramas aprobada/rechazada/integridad
    prod = next(s for s in steps(CI["jobs"]["docker"]) if "APPROVED" in s.get("env", {}))
    assert prod["env"]["APPROVED"] == "${{ needs.test.outputs.approved }}"


def test_jobs_that_decide_fetch_the_full_history():
    """release decide comprueba el historial (registro de test y congelación): sin fetch-depth: 0
    un checkout superficial no lo tendría y la decisión fallaría por integridad."""
    jobs = [CI["jobs"]["test"], CD["jobs"]["gate"], CD["jobs"]["publish"]]
    for job in jobs:
        assert "release decide" in runs(job)
        checkout = next(s for s in steps(job) if s.get("uses", "").startswith("actions/checkout"))
        assert checkout["with"]["fetch-depth"] == 0


def test_actions_are_pinned_to_full_commit_sha():
    for wf in (CI, CD):
        for job in wf["jobs"].values():
            for s in steps(job):
                if "uses" in s:
                    assert re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", s["uses"]), s["uses"]


@pytest.mark.skipif(_bash() is None, reason="bash no disponible")
@pytest.mark.skipif(not is_repository_root(REPO_ROOT), reason="la carpeta aún no es un repositorio Git")
def test_cd_gate_script_blocks_the_real_rejected_model(tmp_path):
    out_file = tmp_path / "github_output"
    out_file.write_text("", encoding="utf-8")
    script = _gate_script().replace("python ", f'"{sys.executable}" ')
    env = {**os.environ, "GITHUB_OUTPUT": str(out_file), "PYTHONPATH": str(REPO_ROOT / "src")}
    r = subprocess.run([_bash(), "-c", script], check=False, cwd=tmp_path, env=env, capture_output=True,
                       encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr  # bloqueado no es un error del workflow
    outputs = dict(line.split("=", 1) for line in out_file.read_text(encoding="utf-8").split())
    record = json.loads((REPO_ROOT / "reports/train/test_evaluation.json").read_text(encoding="utf-8"))
    assert outputs["approved"] == str(record["gate_passed"]).lower()  # sigue a la evidencia commiteada
    assert outputs == {"approved": "false", "model_id": "lr-frozen-feda273"}  # candidato actual: rechazado
    assert "Publicación BLOQUEADA" in r.stdout


@pytest.mark.skipif(_bash() is None, reason="bash no disponible")
@pytest.mark.parametrize("passed,code,approved", [(True, 0, "true"), (False, 0, "false"), (None, 1, None)])
def test_cd_gate_script_on_synthetic_repos(tmp_path, passed, code, approved):
    """Repositorio SINTÉTICO (ver test_release): ejercita las tres ramas del script del gate
    sin publicar nada. None = artefacto manipulado → el job falla."""
    from test_release import make_repo
    repo = make_repo(tmp_path / "repo", passed=bool(passed))
    if passed is None:
        (repo / "models/candidate/model.joblib").write_bytes(b"manipulado")
    (repo / "src/hdfs_anomaly").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "src/hdfs_anomaly/release.py", repo / "src/hdfs_anomaly/release.py")
    (repo / "src/hdfs_anomaly/__init__.py").write_text("", encoding="utf-8")
    out_file = tmp_path / "github_output"
    out_file.write_text("", encoding="utf-8")
    env = {**os.environ, "GITHUB_OUTPUT": str(out_file), "PYTHONPATH": str(repo / "src")}
    script = _gate_script().replace("python ", f'"{sys.executable}" ')
    r = subprocess.run([_bash(), "-c", script], check=False, cwd=tmp_path, env=env, capture_output=True,
                       encoding="utf-8", timeout=120)
    assert r.returncode == code, r.stdout + r.stderr
    if approved:
        assert f"approved={approved}" in out_file.read_text(encoding="utf-8")
