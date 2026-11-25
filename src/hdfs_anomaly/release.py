"""Decisión de publicación: quality gate obligatorio del CD.

    python -m hdfs_anomaly.release decide [--output release_decision.json]

Códigos de salida: 0 = publicación permitida, 3 = publicación BLOQUEADA,
1 = fallo de integridad o validación (no hay decisión posible).

La decisión se calcula SOLO a partir de artefactos verificables y commiteados:
models/candidate/{model.joblib,spec.json}, config.toml, reports/data/split_summary.json y
reports/train/test_evaluation.json. No existe ninguna variable de entorno ni argumento que
permita omitirla. Solo usa la biblioteca estándar (el job de CD no necesita dependencias) y
nunca deserializa el modelo.

Procedencia del candidato:
- native: el commit de congelación existe en este repositorio y es el último que modificó
  spec.json. El gate decide la publicación.
- imported: el candidato procede de otro repositorio cuyo historial no está disponible. Sus
  archivos se comprueban contra reports/origin_provenance.json (SHA-256 y commits de origen),
  pero este historial no demuestra que test se evaluara una sola vez, así que la publicación
  queda BLOQUEADA aunque el gate estuviera superado.
Cualquier otra situación (sin Git, commits inexistentes sin manifiesto, manifiesto que no
cuadra) es un fallo de integridad: se bloquea sin decisión.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # raíz del repositorio (o /app en la imagen)
FEATURES = [f"E{i}" for i in range(1, 30)]
FILES = {"model": "models/candidate/model.joblib", "spec": "models/candidate/spec.json",
         "config": "config.toml", "split_summary": "reports/data/split_summary.json",
         "test_record": "reports/train/test_evaluation.json"}
PROVENANCE = "reports/origin_provenance.json"
EXIT_APPROVED, EXIT_INTEGRITY, EXIT_BLOCKED = 0, 1, 3


class IntegrityError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics_from_cm(m: dict) -> dict:
    """Métricas derivables de la matriz de confusión, con las fórmulas y el redondeo de train.evaluate."""
    cm = m["confusion_matrix"]
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    n = tp + fp + fn + tn
    return {"n": n, "positives": tp + fn,
            "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 6) if tp + fn else None,
            "f1": round(2 * tp / (2 * tp + fp + fn), 6) if tp + fp + fn else None,
            "fpr": round(fp / (fp + tn), 6) if fp + tn else None,
            "accuracy": round((tp + tn) / n, 6)}


def check_metrics(m: dict, where: str) -> None:
    bad = [k for k, v in metrics_from_cm(m).items() if m.get(k) != v]
    if bad:
        raise IntegrityError(f"{where}: {bad} no cuadran con su matriz de confusión")


def gate_checks(test: dict, gate: dict) -> dict:
    return {"recall": (test["recall"] or 0) >= gate["min_recall"],
            "precision": test["precision"] >= gate["min_precision"],
            "fpr": test["fpr"] is not None and test["fpr"] <= gate["max_fpr"]}


def gate_reason(checks: dict) -> str:
    names = {"recall": "recall_below_threshold", "precision": "precision_below_threshold",
             "fpr": "fpr_above_threshold"}
    return ",".join(names[k] for k, ok in checks.items() if not ok) or "none"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True, text=True)


def require_committed(root: Path, rel: str) -> None:
    """El archivo debe estar versionado en el repositorio y sin cambios pendientes."""
    if (_git(root, "ls-files", "--error-unmatch", rel).returncode
            or _git(root, "status", "--porcelain", "--", rel).stdout.strip()):
        raise IntegrityError(f"{rel} no está commiteado o tiene cambios sin commitear")


def require_repository_root(root: Path) -> None:
    top = _git(root, "rev-parse", "--show-toplevel")
    if top.returncode or Path(top.stdout.strip()).resolve() != root.resolve():
        raise IntegrityError(f"{root} no es la raíz de un repositorio Git: sin historial no hay decisión posible")


def candidate_history(root: Path, spec: dict, ev: dict) -> dict:
    """Procedencia del candidato: native o imported (ver el docstring del módulo).
    Lanza IntegrityError si no puede demostrarse ninguna de las dos."""
    freeze = ev.get("freeze_commit") or ""
    if freeze and not _git(root, "cat-file", "-e", f"{freeze}^{{commit}}").returncode:
        last = _git(root, "log", "-1", "--format=%H", "--", FILES["spec"]).stdout.strip()
        if last != freeze:
            raise IntegrityError("El commit de congelación registrado no es el último que modificó spec.json")
        return {"mode": "native", "origin": None}

    path = root / PROVENANCE
    if not path.is_file():
        raise IntegrityError("El commit de congelación no existe en este repositorio y no hay manifiesto "
                             f"de procedencia ({PROVENANCE})")
    require_committed(root, PROVENANCE)
    prov = json.loads(path.read_text(encoding="utf-8"))
    origin, files = prov["origin_commits"], prov["files_sha256"]
    expected = {"commit de congelación": (origin["freeze_commit"], freeze),
                "commit de entrenamiento": (origin["training_code_commit"], spec["training_code_commit"]),
                "commit de evaluación": (origin["evaluation_commit"], ev["evaluation_git"]["commit"])}
    bad = [k for k, (got, exp) in expected.items() if got != exp]
    bad += [f"{rel} no figura en el manifiesto" for rel in FILES.values() if rel not in files]
    for rel, digest in files.items():
        p = root / rel
        if not p.is_file() or sha256(p) != digest:
            bad.append(f"{rel} no coincide con el SHA-256 de origen")
    if bad:
        raise IntegrityError(f"El manifiesto de procedencia no corresponde a estos artefactos: {bad}")
    for rel in files:
        require_committed(root, rel)
    return {"mode": "imported", "origin": origin}


def decide(root: Path) -> dict:
    """Devuelve la decisión de publicación. Lanza IntegrityError si cualquier artefacto falta,
    no está commiteado, ha cambiado o es incoherente; en ese caso no hay decisión posible."""
    root = Path(root).resolve()
    paths = {k: root / v for k, v in FILES.items()}
    missing = [v for k, v in FILES.items() if not paths[k].is_file()]
    if missing:
        raise IntegrityError(f"Faltan artefactos: {missing}")
    require_repository_root(root)
    for rel in FILES.values():
        require_committed(root, rel)
    # En este repositorio, el registro de test no se ha modificado desde que se añadió.
    history = _git(root, "log", "--all", "--format=%H", "--", FILES["test_record"]).stdout.split()
    if len(history) != 1:
        raise IntegrityError(f"El registro de test debe proceder de un único commit (aparece en {len(history)})")

    spec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    ev = json.loads(paths["test_record"].read_text(encoding="utf-8"))
    summary = json.loads(paths["split_summary"].read_text(encoding="utf-8"))
    model_sha, spec_sha = sha256(paths["model"]), sha256(paths["spec"])
    expected = {
        "especificación congelada": (str(spec.get("status", "")).startswith("CONGELADA"), True),
        "características E1–E29 en orden": (spec.get("features"), FEATURES),
        "SHA-256 del modelo": (model_sha, spec["model_sha256"]),
        "SHA-256 de config.toml": (sha256(paths["config"]), spec["config_sha256"]),
        "registro de test terminado": (ev.get("status"), "EVALUADO"),
        "modelo evaluado": (ev["integrity_verified"]["model_sha256"], model_sha),
        "especificación evaluada": (ev["frozen_spec_sha256"], spec_sha),
        "umbral": (ev["threshold"], spec["threshold"]),
        "criterios del gate": (ev["gate"], spec["gate"]),
        "commit de entrenamiento": (ev["training_code_commit"], spec["training_code_commit"]),
        "manifiesto": (ev["integrity_verified"]["manifest_sha256"], summary["manifest_sha256_uncompressed"]),
        "filas de test": (ev["test"]["n"], summary["splits"]["test"]["rows"]),
        "anomalías de test": (ev["test"]["positives"], summary["splits"]["test"]["anomalies"]),
    }
    bad = [k for k, (got, exp) in expected.items() if got != exp]
    if bad:
        raise IntegrityError(f"Artefactos incoherentes: {bad}")
    check_metrics(ev["test"], "test")
    check_metrics(spec["validation_metrics"], "validation")
    checks = gate_checks(ev["test"], spec["gate"])
    if checks != ev["gate_checks"] or all(checks.values()) != ev["gate_passed"]:
        raise IntegrityError("La decisión del gate registrada no coincide con sus métricas")
    hist = candidate_history(root, spec, ev)

    gate_passed = all(checks.values())
    blockers = ([] if gate_passed else ["gate_failed"]) + \
        (["origin_history_not_verifiable"] if hist["mode"] == "imported" else [])
    approved = not blockers
    origin = hist["origin"] or {}
    return {"decision": "publish_allowed" if approved else "blocked", "approved": approved,
            "gate": "passed" if gate_passed else "failed", "reason": gate_reason(checks),
            "blockers": blockers, "provenance": hist["mode"],
            "model_id": f"{spec['family']}-frozen-{ev['freeze_commit'][:7]}", "model_sha256": model_sha,
            "criteria": spec["gate"], "gate_checks": checks,
            "test_metrics": {k: ev["test"][k] for k in ("precision", "recall", "f1", "fpr", "pr_auc")},
            "training_code_commit": spec["training_code_commit"], "freeze_commit": ev["freeze_commit"],
            "test_record_commit": origin.get("test_record_commit", history[0]),
            "test_record_commit_in_this_repository": history[0],
            "evidence_sha256": {rel: sha256(paths[k]) for k, rel in FILES.items()},
            "aliases_allowed": approved,
            "note": "Solo una versión con gate=passed y procedencia verificable en este repositorio "
                    "puede publicarse o recibir challenger/champion."}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["decide"])
    ap.add_argument("--output", default=None, help="escribe la decisión en este JSON")
    args = ap.parse_args(argv)  # sin opciones que alteren la decisión: no hay forma de omitir el gate
    try:
        out = decide(ROOT)
    except (IntegrityError, KeyError, ValueError, TypeError) as e:
        out = {"decision": "integrity_error", "approved": False, "aliases_allowed": False, "error": str(e)}
        code = EXIT_INTEGRITY
    else:
        code = EXIT_APPROVED if out["approved"] else EXIT_BLOCKED
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
