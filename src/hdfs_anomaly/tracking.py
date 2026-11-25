"""Registro en MLflow (SQLite local) del experimento YA realizado, sin reentrenar ni
reevaluar test. Todo se importa de los JSON versionados, conservando la procedencia.

Si el candidato procede de otro repositorio (procedencia «imported», ver release.py), los
commits de origen se toman del manifiesto reports/origin_provenance.json, verificado por
SHA-256, y el run lo indica con la etiqueta provenance.history=imported.

Comandos:
    python -m hdfs_anomaly.tracking register   # importa experimento, run y versión (idempotente)
    python -m hdfs_anomaly.tracking verify     # comprueba MLflow contra los archivos congelados
    python -m hdfs_anomaly.tracking set-alias --alias challenger --version N  # solo si gate=passed

El modelo NUNCA se deserializa aquí: se registra el archivo original model.joblib y se
compara por SHA-256. No se asignan alias (champion/challenger exigen superar el gate).
"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

from mlflow.artifacts import download_artifacts
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

from hdfs_anomaly import release
from hdfs_anomaly import train as T
from hdfs_anomaly.config import DEFAULT_CONFIG, load_config
from hdfs_anomaly.release import IntegrityError, gate_checks, gate_reason
from hdfs_anomaly.release import check_metrics as _check_metrics

# No van en config.toml: ese archivo forma parte del candidato congelado (su SHA-256 consta en spec.json).
EXPERIMENT = "hdfs-anomaly-detection"
MODEL_NAME = "hdfs-anomaly-detector"
PROTECTED_ALIASES = ("champion", "challenger")  # solo para versiones que superan el gate
METRIC_KEYS = ("n", "positives", "pr_auc", "roc_auc", "precision", "recall", "f1", "fpr", "mcc", "accuracy")
# Informes que se adjuntan al run; los marcados con True son obligatorios.
REPORTS = {"data/split_summary.json": True, "data/validation_report.json": False,
           "train/selection_results.json": True, "train/candidate_spec.json": True,
           "train/pattern_diagnostics.json": True, "train/comparativa_validacion.md": False,
           "train/error_analysis.json": False, "train/sensitivity.json": False, "train/rf_pilot.json": False,
           "train/test_evaluation.json": True, "train/test_error_analysis.json": False}


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def client_for(tracking_dir: Path) -> MlflowClient:
    uri = f"sqlite:///{(tracking_dir / 'mlflow.db').resolve().as_posix()}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


# --- Carga y comprobaciones de integridad (antes de escribir nada en MLflow) -----------------
def load_evidence(cfg: dict) -> dict:
    """Lee la especificación congelada y los informes y comprueba que son íntegros, que están
    commiteados sin cambios y que test procede del registro único. Lanza IntegrityError si no."""
    root = cfg["_path"].resolve().parent
    release.require_repository_root(root)
    mdir, reports = cfg["output"]["models"], cfg["output"]["reports"].parent
    spec_path, model_path = mdir / "spec.json", mdir / "model.joblib"
    files = {"model/spec.json": spec_path, "model/model.joblib": model_path}
    for rel, required in REPORTS.items():
        p = reports / rel
        if p.is_file():
            files[f"reports/{rel}"] = p
        elif required:
            raise IntegrityError(f"Falta el informe obligatorio {p.relative_to(root).as_posix()}")

    git = {}
    for name, p in files.items():
        rel = p.relative_to(root).as_posix()
        if (T._git(root, "ls-files", "--error-unmatch", rel).returncode
                or T._git(root, "status", "--porcelain", "--", rel).stdout.strip()):
            raise IntegrityError(f"{rel} no está commiteado o tiene cambios sin commitear")
        git[name] = {"path": rel, "sha256": T._sha(p),
                     "commit": T._git(root, "log", "-1", "--format=%H", "--", rel).stdout.strip()}

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != T.FROZEN:
        raise IntegrityError("models/candidate/spec.json no está congelada")
    if git["model/model.joblib"]["sha256"] != spec["model_sha256"]:
        raise IntegrityError("model.joblib no coincide con el SHA-256 de la especificación congelada")
    if T._sha(cfg["_path"]) != spec["config_sha256"]:
        raise IntegrityError("config.toml ha cambiado desde la congelación")

    te_rel = git["reports/train/test_evaluation.json"]["path"]
    history = T._git(root, "log", "--all", "--format=%H", "--", te_rel).stdout.split()
    if len(history) != 1:
        raise IntegrityError(f"{te_rel} debe proceder de un único commit y no haberse modificado "
                             f"(aparece en {len(history)} commits)")
    ev = json.loads(files["reports/train/test_evaluation.json"].read_text(encoding="utf-8"))
    hist = release.candidate_history(root, spec, ev)
    if hist["mode"] == "imported":
        files["reports/origin_provenance.json"] = root / release.PROVENANCE
        git["reports/origin_provenance.json"] = {
            "path": release.PROVENANCE, "sha256": T._sha(root / release.PROVENANCE),
            "commit": T._git(root, "log", "-1", "--format=%H", "--", release.PROVENANCE).stdout.strip()}
    summary = json.loads(files["reports/data/split_summary.json"].read_text(encoding="utf-8"))
    pattern = json.loads(files["reports/train/pattern_diagnostics.json"].read_text(encoding="utf-8"))
    val_su = pattern["validation_seen_unseen"][spec["family"]]

    expected = {
        "evaluación terminada": (ev.get("status"), "EVALUADO"),
        "SHA-256 del modelo evaluado": (ev["integrity_verified"]["model_sha256"], spec["model_sha256"]),
        "SHA-256 de config evaluada": (ev["integrity_verified"]["config_sha256"], spec["config_sha256"]),
        "SHA-256 del manifiesto": (ev["integrity_verified"]["manifest_sha256"], spec["manifest_sha256"]),
        "SHA-256 de la especificación evaluada": (ev["frozen_spec_sha256"], git["model/spec.json"]["sha256"]),
        "commit de entrenamiento": (ev["training_code_commit"], spec["training_code_commit"]),
        "umbral": (ev["threshold"], spec["threshold"]),
        "gate": (ev["gate"], spec["gate"]),
        "manifiesto de split_summary": (summary["manifest_sha256_uncompressed"], spec["manifest_sha256"]),
        "filas de test": (ev["test"]["n"], summary["splits"]["test"]["rows"]),
        "anomalías de test": (ev["test"]["positives"], summary["splits"]["test"]["anomalies"]),
        "filas de validation": (spec["validation_metrics"]["n"], summary["splits"]["validation"]["rows"]),
        "test visto + no visto": (sum(g["n"] for g in ev["test_seen_unseen"].values()), ev["test"]["n"]),
        "validation de pattern_diagnostics": (val_su["threshold"], spec["threshold"]),
    }
    bad = [k for k, (got, exp) in expected.items() if got != exp]
    if bad:
        raise IntegrityError(f"Informes incoherentes con la especificación congelada: {bad}")

    groups = {"validation": spec["validation_metrics"], "test": ev["test"],
              "validation_seen": val_su["seen_in_train"], "validation_unseen": val_su["unseen_in_train"],
              "test_seen": ev["test_seen_unseen"]["seen_in_train"],
              "test_unseen": ev["test_seen_unseen"]["unseen_in_train"]}
    for name, m in groups.items():
        if "confusion_matrix" in m:
            _check_metrics(m, name)
    g, t = spec["gate"], ev["test"]
    checks = gate_checks(t, g)
    if checks != ev["gate_checks"] or all(checks.values()) != ev["gate_passed"]:
        raise IntegrityError("La decisión del gate registrada no coincide con las métricas de test")

    provenance = {
        "validation": f"{git['model/spec.json']['path']}#validation_metrics",
        "validation_seen": f"{git['reports/train/pattern_diagnostics.json']['path']}"
                           f"#validation_seen_unseen.{spec['family']}.seen_in_train",
        "validation_unseen": f"{git['reports/train/pattern_diagnostics.json']['path']}"
                             f"#validation_seen_unseen.{spec['family']}.unseen_in_train",
        "test": f"{te_rel}#test", "test_seen": f"{te_rel}#test_seen_unseen.seen_in_train",
        "test_unseen": f"{te_rel}#test_seen_unseen.unseen_in_train"}
    import_key = _sha_bytes((spec["model_sha256"] + git["model/spec.json"]["sha256"]
                             + git["reports/train/test_evaluation.json"]["sha256"]).encode())
    test_record_commit = hist["origin"]["test_record_commit"] if hist["origin"] else history[0]
    return {"root": root, "spec": spec, "ev": ev, "files": files, "git": git, "groups": groups,
            "gate_checks": checks, "provenance": provenance, "import_key": import_key,
            "test_record_commit": test_record_commit, "history": hist["mode"]}


# --- Registro ---------------------------------------------------------------------------
def _status(ev_: dict) -> dict:
    passed = ev_["ev"]["gate_passed"]
    return {"gate": "passed" if passed else "failed", "status": "gate_passed" if passed else "rejected",
            "reason": gate_reason(ev_["gate_checks"]), "test_evaluated": "true",
            "promotion": "not_requested" if passed else "blocked"}


def _run_contents(e: dict) -> tuple[list, list, list]:
    spec, ev = e["spec"], e["ev"]
    clf = T.build_model(spec["family"], spec["params"], spec["seed"]).named_steps["clf"]  # sin ajustar
    params = {"family": spec["family"], **{f"model.{k}": v for k, v in spec["params"].items()},
              "model.max_iter": getattr(clf, "max_iter", None), "pipeline": "log1p -> StandardScaler -> LogisticRegression"
              if spec["family"] == "lr" else spec["family"], "seed": spec["seed"],
              "threshold": repr(spec["threshold"]), "threshold_rule": spec["threshold_rule"],
              "partition": "completion", "trained_on": spec["trained_on"], "n_features": len(spec["features"]),
              **{f"gate.{k}": v for k, v in spec["gate"].items()}}
    env = spec["environment"]
    tags = {**_status(e),
            "training_code_commit": spec["training_code_commit"], "freeze_commit": ev["freeze_commit"],
            "evaluation_commit": ev["evaluation_git"]["commit"], "test_record_commit": e["test_record_commit"],
            "mlflow.source.git.commit": spec["training_code_commit"],
            "python_version": env["python"], "sklearn_version": env["sklearn"],
            "pandas_version": env["pandas"], "numpy_version": env["numpy"],
            "sha256.model": spec["model_sha256"], "sha256.config": spec["config_sha256"],
            "sha256.manifest": spec["manifest_sha256"], "sha256.frozen_spec": e["git"]["model/spec.json"]["sha256"],
            "sha256.test_evaluation": e["git"]["reports/train/test_evaluation.json"]["sha256"],
            **{f"sha256.data.{name}": v["sha256"] for name, v in spec["sources"].items()},
            **{f"provenance.{k}": v for k, v in e["provenance"].items()},
            "import_key": e["import_key"], "imported_without_retraining": "true",
            "provenance.history": e["history"]}
    ts = int(time.time() * 1000)
    metrics = []
    for group, m in e["groups"].items():
        metrics += [Metric(f"{group}/{k}", float(m[k]), ts, 0) for k in METRIC_KEYS if m.get(k) is not None]
        metrics += [Metric(f"{group}/{k}", float(v), ts, 0) for k, v in m.get("confusion_matrix", {}).items()]
    return ([Param(k, str(v)) for k, v in params.items()], [RunTag(k, str(v)) for k, v in tags.items()], metrics)


def _find_run(client: MlflowClient, exp_id: str, key: str) -> list:
    return client.search_runs([exp_id], f"tags.import_key = '{key}'")


def register(cfg: dict, tracking_dir: Path) -> dict:
    """Importa el experimento ya realizado. Idempotente: si ya existe un run completo con la
    misma import_key (modelo + especificación + registro de test), no crea nada nuevo."""
    e = load_evidence(cfg)
    client = client_for(tracking_dir)
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(
        EXPERIMENT, artifact_location=(tracking_dir / "mlartifacts").resolve().as_uri(),
        tags={"project": "hdfs_anomaly", "dataset": "Loghub HDFS_v1", "phase": "4"})

    runs = _find_run(client, exp_id, e["import_key"])
    for r in runs:  # importación anterior interrumpida: se descarta y se repite
        if r.data.tags.get("import_complete") != "true":
            client.delete_run(r.info.run_id)
    runs = [r for r in runs if r.data.tags.get("import_complete") == "true"]
    if len(runs) > 1:
        raise IntegrityError(f"Hay {len(runs)} runs completos con la misma import_key")
    created_run = not runs
    if created_run:
        spec = e["spec"]
        run = client.create_run(exp_id, run_name=f"{spec['family']}-frozen-{e['ev']['freeze_commit'][:7]}")
        rid = run.info.run_id
        params, tags, metrics = _run_contents(e)
        root = e["root"]
        tags += [RunTag("importer.git_commit", T._git(root, "rev-parse", "HEAD").stdout.strip()),
                 RunTag("importer.git_dirty", str(bool(T._git(root, "status", "--porcelain").stdout.strip())).lower())]
        client.log_batch(rid, metrics=metrics, params=params, tags=tags)
        for name, p in e["files"].items():
            client.log_artifact(rid, str(p), str(Path(name).parent).replace("\\", "/"))
        with tempfile.TemporaryDirectory() as d:
            prov = Path(d) / "provenance.json"
            prov.write_text(json.dumps({"metrics": e["provenance"], "files": e["git"],
                                        "test_record_commit": e["test_record_commit"]}, indent=2), encoding="utf-8")
            client.log_artifact(rid, str(prov))
        client.set_tag(rid, "import_complete", "true")
        client.set_terminated(rid)
    else:
        rid = runs[0].info.run_id

    if not client.search_registered_models(f"name = '{MODEL_NAME}'"):
        client.create_registered_model(
            MODEL_NAME, tags={"task": "binary_classification", "dataset": "Loghub HDFS_v1"},
            description="Detector de trazas HDFS anómalas. Solo pasan a champion/challenger las versiones "
                        "que superan el quality gate.")
    versions = [v for v in client.search_model_versions(f"name = '{MODEL_NAME}'") if v.run_id == rid]
    if len(versions) > 1:
        raise IntegrityError(f"Hay {len(versions)} versiones registradas para el mismo run")
    created_version = not versions
    if created_version:
        st = _status(e)
        mv = client.create_model_version(
            MODEL_NAME, f"runs:/{rid}/model", run_id=rid,
            tags={**st, "model_sha256": e["spec"]["model_sha256"],
                  "training_code_commit": e["spec"]["training_code_commit"],
                  "freeze_commit": e["ev"]["freeze_commit"], "artifact_format": "joblib (archivo original)"},
            description=f"Candidato congelado {e['spec']['family']} {e['spec']['params']}; "
                        f"gate {st['gate']} ({st['reason']}). No promocionable.")
        version = mv.version
    else:
        version = versions[0].version
    return {"experiment_id": exp_id, "run_id": rid, "model_name": MODEL_NAME, "version": str(version),
            "created_run": created_run, "created_version": created_version, **_status(e)}


def verify(cfg: dict, tracking_dir: Path) -> dict:
    """Comprueba que lo registrado en MLflow coincide con los archivos congelados: un único run
    y una única versión, el artefacto idéntico por SHA-256 (sin deserializarlo), las métricas
    iguales a las de los JSON y ningún alias protegido sobre una versión que no superó el gate."""
    e = load_evidence(cfg)
    client = client_for(tracking_dir)
    exp = client.get_experiment_by_name(EXPERIMENT)
    if not exp:
        raise IntegrityError(f"No existe el experimento {EXPERIMENT}")
    runs = _find_run(client, exp.experiment_id, e["import_key"])
    if len(runs) != 1 or runs[0].data.tags.get("import_complete") != "true":
        raise IntegrityError(f"Se esperaba un run completo con la import_key; hay {len(runs)}")
    run = runs[0]
    _, tags, metrics = _run_contents(e)
    want_tags = {t.key: t.value for t in tags}
    bad = [k for k, v in want_tags.items() if run.data.tags.get(k) != v]
    bad += [m.key for m in metrics if run.data.metrics.get(m.key) != m.value]
    if bad:
        raise IntegrityError(f"MLflow no coincide con los informes en: {bad}")
    versions = [v for v in client.search_model_versions(f"name = '{MODEL_NAME}'") if v.run_id == run.info.run_id]
    if len(versions) != 1:
        raise IntegrityError(f"Se esperaba una versión registrada para el run; hay {len(versions)}")
    mv = versions[0]
    with tempfile.TemporaryDirectory() as d:
        local = download_artifacts(artifact_uri=f"{run.info.artifact_uri}/model/model.joblib", dst_path=d)
        registered_sha = T._sha(Path(local))  # se compara el archivo, nunca se deserializa
    if not registered_sha == mv.tags.get("model_sha256") == e["spec"]["model_sha256"] \
            == e["git"]["model/model.joblib"]["sha256"]:
        raise IntegrityError("El artefacto registrado no es idéntico al candidato congelado")
    if mv.tags.get("status") != want_tags["status"] or mv.tags.get("gate") != want_tags["gate"]:
        raise IntegrityError("Las etiquetas de la versión registrada no reflejan el gate")
    aliases = client.get_registered_model(MODEL_NAME).aliases
    misuse = {a: v for a, v in aliases.items() if a in PROTECTED_ALIASES
              and client.get_model_version(MODEL_NAME, v).tags.get("gate") != "passed"}
    if misuse:
        raise IntegrityError(f"Alias protegidos sobre versiones que no superaron el gate: {misuse}")
    return {"run_id": run.info.run_id, "version": str(mv.version), "model_sha256": registered_sha,
            "metrics_checked": len(metrics), "tags_checked": len(want_tags), "aliases": aliases,
            "status": mv.tags.get("status"), "gate": mv.tags.get("gate"), "reason": mv.tags.get("reason")}


def set_alias(cfg: dict, tracking_dir: Path, alias: str, version: str) -> dict:
    """Única vía prevista para asignar challenger/champion. Exige que la versión tenga
    gate=passed y que la decisión de publicación (release.decide) sobre los artefactos del
    repositorio sea favorable y corresponda al mismo SHA-256. Un modelo rechazado nunca pasa."""
    if alias not in PROTECTED_ALIASES:
        raise IntegrityError(f"Alias no previsto: {alias}")
    client = client_for(tracking_dir)
    mv = client.get_model_version(MODEL_NAME, version)
    if mv.tags.get("gate") != "passed" or mv.tags.get("status") == "rejected":
        raise IntegrityError(f"La versión {version} no superó el quality gate (gate={mv.tags.get('gate')}, "
                             f"status={mv.tags.get('status')}): no puede recibir '{alias}'")
    decision = release.decide(cfg["_path"].resolve().parent)
    if not decision["approved"] or decision["model_sha256"] != mv.tags.get("model_sha256"):
        raise IntegrityError("La decisión de publicación no autoriza esta versión")
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    return {"alias": alias, "version": version, "model_sha256": decision["model_sha256"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["register", "verify", "set-alias"])
    ap.add_argument("--alias", choices=PROTECTED_ALIASES, help="set-alias: challenger o champion")
    ap.add_argument("--version", help="set-alias: versión del Model Registry")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tracking-dir", default=None, help="carpeta de mlflow.db y mlartifacts/ (def.: raíz)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cfg["_path"] = Path(args.config) if args.config else DEFAULT_CONFIG
    tracking_dir = Path(args.tracking_dir) if args.tracking_dir else cfg["_path"].resolve().parent
    try:
        if args.command == "set-alias":
            out = set_alias(cfg, tracking_dir, args.alias, args.version)
        else:
            out = register(cfg, tracking_dir) if args.command == "register" else verify(cfg, tracking_dir)
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
