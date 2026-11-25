# HDFS anomaly detection — pipeline MLOps

Detector de trazas de bloque HDFS anómalas. El pipeline cubre:
- validación y partición cronológica de los datos;
- entrenamiento y selección del modelo;
- congelación del candidato y quality gate;
- registro en MLflow;
- API REST con FastAPI e imagen Docker;
- CI/CD con GitHub Actions;
- monitorización: métricas Prometheus y deriva con PSI.

## Estado del modelo

> **El único candidato incluido (`models/candidate/`) está RECHAZADO por el quality gate.** No hay despliegue productivo.

| | |
|---|---|
| Modelo | `log1p → StandardScaler → LogisticRegression` (C=10, `class_weight=None`, `max_iter=2000`, semilla 42), 29 variables E1–E29 |
| Umbral | `0.07222539011103471` |
| SHA-256 | `da81e6ba7be50c103887e78a5912340cd40f563d9c20368827f55402b79d1076` |

| Métrica en test | Valor | Criterio | |
|---|---:|---:|---|
| Precision | 0,8865 | ≥ 0,90 | **No cumple** |
| Recall | 0,9978 | ≥ 0,90 | Cumple |
| FPR | 0,305 % | ≤ 0,5 % | Cumple |

Estado del candidato: `gate=failed`, `status=rejected`, `reason=precision_below_threshold`, `promotion=blocked`. Esto implica:
- **Arranque:** la API y la imagen Docker solo arrancan con `APP_MODE=demo`. Con `APP_MODE=production`, con un modo desconocido o sin `APP_MODE` el arranque se rechaza.
- **Publicación:** el workflow de CD la bloquea.
- **MLflow:** el modelo no puede recibir los alias `challenger` ni `champion`.
- **Test:** ya se evaluó y **no debe repetirse**. `train evaluate-test` se niega a ejecutarse porque `reports/train/test_evaluation.json` ya existe.

## Requisitos

- **Python 3.14.7** (`.python-version`).
- **Git.** `release decide` y el registro en MLflow comprueban el historial del repositorio.
- **Docker con Docker Compose**, solo para la imagen y sus pruebas.
- **Dependencias bloqueadas:**

| Archivo | Uso |
|---|---|
| `requirements-dev.lock.txt` | desarrollo, pruebas y CI (bloqueo completo) |
| `requirements-api.txt` | imagen Docker (bloqueo completo, sin MLflow) |
| `requirements.txt`, `requirements-api.in` | dependencias directas a partir de las que se generan los bloqueos |

## Instalación

Linux / macOS:

```bash
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.lock.txt
python -m pip install --no-deps -e .
```

Windows (PowerShell):

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock.txt
python -m pip install --no-deps -e .
```

Comprobación:

```bash
ruff check .
python -m pytest
```

## Dataset HDFS_v1 (externo)

El dataset no se incluye. Se descarga de [Loghub](https://github.com/logpai/loghub) (HDFS_v1) y se descomprime fuera del repositorio o en `data/HDFS_v1`, carpeta ignorada por Git.

Su ubicación se indica con `HDFS_V1_DIR` o con `--data-dir`:

```bash
export HDFS_V1_DIR=/ruta/a/HDFS_v1          # Linux / macOS
```

```powershell
$env:HDFS_V1_DIR = "C:\ruta\a\HDFS_v1"      # PowerShell
```

La carpeta debe contener `preprocessed/Event_occurrence_matrix.csv`, `preprocessed/anomaly_label.csv` y `HDFS.log`.

| Requiere el dataset | Funciona sin él |
|---|---|
| `python -m hdfs_anomaly.timeline` (línea temporal desde `HDFS.log`) | API y Docker |
| `python -m hdfs_anomaly.data` (validación y partición) | `release decide` |
| `python -m hdfs_anomaly.train select` (entrenamiento y selección) | registro y verificación en MLflow |
| `python -m hdfs_anomaly.drift reference` (referencia de deriva) | `simulate` y `drift report` |
| `python tests/fixtures/make_sample.py` (muestra de pruebas) | pruebas ordinarias |

Las pruebas usan `tests/fixtures/`, una muestra de 2000 bloques tomada solo de la partición train.

## Uso

### API local (modo demostración)

```bash
APP_MODE=demo uvicorn hdfs_anomaly.api:create_app --factory --host 127.0.0.1 --port 8000
```

```powershell
$env:APP_MODE = "demo"; uvicorn hdfs_anomaly.api:create_app --factory --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model/info
curl http://127.0.0.1:8000/metrics
curl -X POST http://127.0.0.1:8000/predict -H "content-type: application/json" \
  -d '{"E1":0,"E2":0,"E3":0,"E4":0,"E5":2,"E6":0,"E7":1,"E8":0,"E9":0,"E10":0,"E11":0,"E12":0,"E13":0,"E14":0,"E15":0,"E16":0,"E17":0,"E18":0,"E19":0,"E20":0,"E21":0,"E22":1,"E23":0,"E24":0,"E25":0,"E26":0,"E27":0,"E28":0,"E29":0}'
```

- **Entrada de `/predict`:**
  - exige las 29 variables E1–E29 como enteros ≥ 0;
  - rechaza cualquier otro campo (`BlockId`, `Label`, `Type`…) con 422.
- **Respuesta:** la clase, la puntuación, el umbral, la identidad del modelo, el estado del gate y `demo_only=true`.

### Docker

```bash
docker build -t hdfs-anomaly:demo-local .
docker run --rm -e APP_MODE=demo -p 127.0.0.1:8000:8000 hdfs-anomaly:demo-local
```

Con Compose (perfil `demo`, puerto solo en 127.0.0.1, sistema de archivos de solo lectura):

```bash
docker compose --profile demo up --build -d
docker compose --profile demo down
```

Pruebas de humo de la imagen (requieren la imagen construida; `DOCKER_IMAGE` permite usar otra etiqueta):

```bash
RUN_DOCKER_TESTS=1 python -m pytest tests/test_api.py -k docker
```

### Monitorización

Con la API en marcha:

```bash
python -m hdfs_anomaly.simulate --base-url http://127.0.0.1:8000
python -m hdfs_anomaly.drift report reports/monitoring/windows/shifted.csv --output informe.json
```

- **`simulate`:**
  - envía ventanas generadas desde la referencia de train (`monitoring/reference_train.json`) y perturbaciones sintéticas;
  - nunca usa test;
  - escribe las ventanas, sus informes de deriva y una instantánea de `/metrics` en `reports/monitoring/`.
- **Interpretación:** un PSI alto indica un cambio de distribución, no una degradación demostrada. Para medir la degradación hacen falta etiquetas reales posteriores.

### Quality gate

```bash
python -m hdfs_anomaly.release decide --output release_decision.json
```

Códigos de salida: `0` publicación permitida, `3` bloqueada, `1` fallo de integridad.
- **Con el candidato incluido:** el resultado esperado es `3` cuando la carpeta es un repositorio Git con los archivos commiteados.
- **Fuera de un repositorio Git:** devuelve `1`.

### MLflow (SQLite local, sin servidor)

```bash
python -m hdfs_anomaly.tracking register
python -m hdfs_anomaly.tracking verify
mlflow ui --backend-store-uri sqlite:///mlflow.db   # opcional
```

- **Qué hace:** importa el experimento ya realizado, sin reentrenar ni volver a evaluar test.
- **Dónde guarda:** crea `mlflow.db` y `mlartifacts/` en la raíz, ignorados por Git; `--tracking-dir RUTA` usa otra carpeta.
- **Requisito:** la carpeta debe ser un repositorio Git con los artefactos commiteados.
- **Resultado:** el registro es idempotente y la versión queda con `gate=failed`, `status=rejected`, `promotion=blocked` y sin alias.

### Pruebas

```bash
python -m pytest                                              # sin dataset ni Docker
python -m pytest tests/test_api.py                            # contrato de la API y /metrics
python -m pytest tests/test_release.py tests/test_workflows.py   # gate y workflows
```

Se omiten, indicando el motivo:
- las de Docker, si no se define `RUN_DOCKER_TESTS=1`;
- las que necesitan el dataset completo;
- las del candidato real, si la carpeta aún no es un repositorio Git.

## CI/CD (GitHub Actions)

| Workflow | Cuándo | Qué hace |
|---|---|---|
| `ci.yml` | cada push y pull request | Ejecuta `ruff`, `pytest`, el contrato de la API y la decisión del gate: aprobado o bloqueado son resultados válidos y solo falla la integridad. Construye la imagen y hace una prueba de humo. No usa dataset ni secretos y no publica nada |
| `cd.yml` | push a `main` | Ejecuta `release decide`. Solo si aprueba, vuelve a verificar, construye y publica en GHCR. **Con el candidato incluido no publica** |

Las acciones están fijadas por SHA de commit y los permisos son los mínimos. Los jobs que deciden hacen checkout con `fetch-depth: 0`, porque el gate comprueba el historial.

## Procedencia e integridad

El candidato se entrenó, congeló y evaluó en un repositorio anterior cuyo historial no forma parte de este. `reports/origin_provenance.json` registra:
- los commits de origen:
  - entrenamiento `a869153`;
  - congelación `feda273`;
  - evaluación `4bdec59`;
  - registro de test `0f05be6`;
  - instantánea copiada `bad22d6`;
- el SHA-256 de cada artefacto e informe tal como estaban en ese repositorio.

**Se verifica aquí:**
- el SHA-256 del modelo, de `spec.json`, de `config.toml` y de los informes;
- la coherencia de cada métrica con su matriz de confusión;
- que la decisión del gate coincide con las métricas y que los criterios no cambiaron;
- que los artefactos están commiteados sin cambios;
- que en este repositorio el registro de test no se ha modificado desde que se añadió.

**No puede verificarse aquí:**
- que test se evaluara una sola vez;
- el orden entrenamiento → congelación → evaluación.

Esas garantías dependían del historial de origen; un SHA-256 comprueba la integridad de un archivo, no cuántas veces se ejecutó una evaluación. Por eso `release decide` trata el candidato como `imported` y bloquea su publicación aunque superara el gate: `blockers: gate_failed, origin_history_not_verifiable`. En MLflow, el run queda con `provenance.history=imported`.

### Publicar un modelo nuevo

Hace falta un candidato cuyo ciclo completo quede en el historial del repositorio donde se publica:
1. `train select` desde un commit limpio.
2. `train freeze` y commit de `models/candidate/`.
3. `train evaluate-test` una sola vez, con datos de test nuevos, y commit del registro.
4. El gate debe superarse. `release decide` lo reconocerá como `native` y el CD podrá publicar.

**Limitación actual:**
- **Por qué no se puede hoy:** `evaluate-test` no se ejecuta si `reports/train/test_evaluation.json` existe o ha existido en el historial, y en este repositorio esa ruta figura desde el primer commit.
- **Qué hace falta:** un repositorio nuevo sin el candidato actual, o un cambio de diseño con registros de test por experimento.
- **Mientras tanto:** el CD permanece bloqueado.

## Atribución de datos

Los datos de `tests/fixtures/` derivan de HDFS_v1 de [Loghub](https://github.com/logpai/loghub) (CC BY 4.0).
- Zhu, J. et al. (2023). *Loghub: A large collection of system log datasets for AI-driven log analytics*. ISSRE 2023.
- Xu, W. et al. (2009). *Detecting large-scale system problems by mining console logs*. SOSP 2009.
