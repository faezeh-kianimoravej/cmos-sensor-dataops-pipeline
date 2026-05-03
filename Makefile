SHELL := /bin/sh

ifneq ($(wildcard .env),)
include .env
export
endif

# PYTHON precedence (explicit + CI-friendly):
# 1) Caller-provided PYTHON wins (e.g. make PYTHON=python pipeline-run)
# 2) Otherwise use local .venv interpreter when available
# 3) Otherwise fall back to system python
ifeq ($(origin PYTHON), undefined)
ifeq ($(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe)
PYTHON := .venv/Scripts/python.exe
PYTHON_SOURCE := local .venv (Windows)
else ifeq ($(wildcard .venv/bin/python),.venv/bin/python)
PYTHON := .venv/bin/python
PYTHON_SOURCE := local .venv (Unix)
else
PYTHON := python
PYTHON_SOURCE := system default
endif
else
PYTHON_SOURCE := provided by caller
endif

CONFIG ?= configs/default.yaml
INPUT ?= data/raw/CMOS_addtional_datasets
COMPOSE_FILE ?= docker/docker-compose.yml
COMPOSE_ENV_FILE ?= .env
PYTEST_ARGS ?= -q --tb=short
COMPOSE_RUN = docker-compose --env-file $(COMPOSE_ENV_FILE) -f $(COMPOSE_FILE)
RUNTIME_TMP ?= .runtime_tmp

ifeq ($(OS),Windows_NT)
PYTEST_ENV_PREFIX := set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 &&
PYTHON_RUN := $(subst /,\,$(PYTHON))
PYTHON_FLAGS := -X utf8
RUN_ENV_PREFIX := set PYTHONIOENCODING=utf-8 && set TMPDIR=$(subst /,\,$(abspath $(RUNTIME_TMP))) && set TMP=$(subst /,\,$(abspath $(RUNTIME_TMP))) && set TEMP=$(subst /,\,$(abspath $(RUNTIME_TMP))) &&
else
PYTEST_ENV_PREFIX := PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
PYTHON_RUN := $(PYTHON)
PYTHON_FLAGS := -X utf8
RUN_ENV_PREFIX := PYTHONIOENCODING=utf-8 TMPDIR=$(abspath $(RUNTIME_TMP)) TMP=$(abspath $(RUNTIME_TMP)) TEMP=$(abspath $(RUNTIME_TMP))
endif

.PHONY: help check-demo-env platform-up platform-down storage-up demo-storage demo-prepare infra-smoke dvc-remote-setup pipeline-run pipeline-train pipeline-eval pipeline-predict api-run airflow-run mlflow-run clean-artifacts test test-unit test-integration test-api test-all

help:
	@echo Developer commands:
	@echo   make pipeline-run
	@echo     full pipeline: ingest -^> features -^> stage1 -^> stage2 subset -^> stage2 -^> evaluate -^> batch
	@echo   make pipeline-train
	@echo     training path only: ingest -^> features -^> stage1 -^> stage2 subset -^> stage2
	@echo   make pipeline-eval
	@echo     evaluate stage1 + stage2 prediction artifacts
	@echo   make pipeline-predict
	@echo     batch prediction only
	@echo   make api-run
	@echo     run FastAPI locally on :8000
	@echo   make airflow-run
	@echo     start Airflow services from docker compose
	@echo   make mlflow-run
	@echo     start MLflow service from docker compose
	@echo   make storage-up
	@echo     start MinIO, bucket bootstrap, and MLflow for the object-storage demo
	@echo   make platform-up
	@echo     start MinIO, MLflow, Airflow, Postgres, and the API together for the demo
	@echo   make platform-down
	@echo     stop the full demo platform stack
	@echo   make check-demo-env
	@echo     verify that .env exists and Docker Desktop is running before the demo
	@echo   make demo-storage
	@echo     run the final storage demo smoke flow
	@echo   make demo-prepare
	@echo     start storage services and prepare train/eval/predict artifacts for the live demo
	@echo   make infra-smoke
	@echo     run a minimal MinIO + MLflow + DVC smoke test
	@echo   make dvc-remote-setup
	@echo     configure local DVC MinIO endpoint in .dvc/config.local from env
	@echo   make clean-artifacts
	@echo     remove generated models/reports/predictions/processed parquet
	@echo   make test
	@echo     run fast default suite (excludes requires_dat_file)
	@echo   make test-unit
	@echo     run unit test layer only
	@echo   make test-integration
	@echo     run integration test layer only
	@echo   make test-api
	@echo     run api test layer only
	@echo   make test-all
	@echo     run all tests including requires_dat_file
	@echo Overridable vars:
	@echo   PYTHON=$(PYTHON)
	@echo   CONFIG=$(CONFIG)
	@echo   INPUT=$(INPUT)
	@echo   COMPOSE_FILE=$(COMPOSE_FILE)
	@echo   PYTEST_ARGS=$(PYTEST_ARGS)
	@echo   DVC_S3_ENDPOINT_URL=$(DVC_S3_ENDPOINT_URL)
	@echo Selected Python source: $(PYTHON_SOURCE)

check-demo-env:
	@$(PYTHON_RUN) -c "from pathlib import Path; import sys; p = Path('.env'); print('Missing .env file. Create it from .env.example before running the demo.') if not p.exists() else None; sys.exit(0 if p.exists() else 1)"
	@$(PYTHON_RUN) -c "import subprocess, sys; result = subprocess.run(['docker', 'version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('Docker Desktop is not running or not reachable. Start Docker Desktop first.') if result.returncode != 0 else None; sys.exit(result.returncode)"

storage-up: check-demo-env
	$(COMPOSE_RUN) up -d minio minio-init mlflow

platform-up: check-demo-env
	$(COMPOSE_RUN) up -d minio minio-init mlflow postgres airflow-init airflow-webserver airflow-scheduler observed-api

platform-down: check-demo-env
	$(COMPOSE_RUN) down

demo-storage: infra-smoke

demo-prepare:
	$(MAKE) check-demo-env
	$(MAKE) storage-up
	$(MAKE) pipeline-train
	$(MAKE) pipeline-eval
	$(MAKE) pipeline-predict

infra-smoke:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	$(PYTHON_RUN) scripts/run_storage_smoke.py --compose-file $(COMPOSE_FILE)

dvc-remote-setup:
	dvc remote modify --local minio endpointurl $(DVC_S3_ENDPOINT_URL)
	dvc remote modify --local minio use_ssl false
	dvc remote modify --local minio access_key_id $${AWS_ACCESS_KEY_ID}
	dvc remote modify --local minio secret_access_key $${AWS_SECRET_ACCESS_KEY}
	dvc remote modify --local minio region $${AWS_DEFAULT_REGION}
	dvc remote modify --local minio listobjects true

pipeline-run:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	@$(PYTHON_RUN) -c "from pathlib import Path; Path(r'$(RUNTIME_TMP)').mkdir(parents=True, exist_ok=True)"
	$(RUN_ENV_PREFIX) $(PYTHON_RUN) $(PYTHON_FLAGS) scripts/run_pipeline.py run --config $(CONFIG) --input $(INPUT)

pipeline-train:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	@$(PYTHON_RUN) -c "from pathlib import Path; Path(r'$(RUNTIME_TMP)').mkdir(parents=True, exist_ok=True)"
	$(RUN_ENV_PREFIX) $(PYTHON_RUN) $(PYTHON_FLAGS) scripts/run_pipeline.py train --config $(CONFIG) --input $(INPUT)

pipeline-eval:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	@$(PYTHON_RUN) -c "from pathlib import Path; Path(r'$(RUNTIME_TMP)').mkdir(parents=True, exist_ok=True)"
	$(RUN_ENV_PREFIX) $(PYTHON_RUN) $(PYTHON_FLAGS) scripts/run_pipeline.py evaluate --config $(CONFIG)

pipeline-predict:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	@$(PYTHON_RUN) -c "from pathlib import Path; Path(r'$(RUNTIME_TMP)').mkdir(parents=True, exist_ok=True)"
	$(RUN_ENV_PREFIX) $(PYTHON_RUN) $(PYTHON_FLAGS) scripts/run_pipeline.py predict --config $(CONFIG)

api-run:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	@$(PYTHON_RUN) -c "from pathlib import Path; Path(r'$(RUNTIME_TMP)').mkdir(parents=True, exist_ok=True)"
	$(RUN_ENV_PREFIX) $(PYTHON_RUN) $(PYTHON_FLAGS) -m api.app

airflow-run:
	$(COMPOSE_RUN) up -d postgres airflow-init airflow-webserver airflow-scheduler

mlflow-run:
	$(COMPOSE_RUN) up -d mlflow

clean-artifacts:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	@$(PYTHON_RUN) -c "from pathlib import Path; Path(r'$(RUNTIME_TMP)').mkdir(parents=True, exist_ok=True)"
	$(RUN_ENV_PREFIX) $(PYTHON_RUN) $(PYTHON_FLAGS) scripts/run_pipeline.py clean

test:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	$(PYTEST_ENV_PREFIX) $(PYTHON_RUN) -m pytest $(PYTEST_ARGS) -m "not requires_dat_file"

test-unit:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	$(PYTEST_ENV_PREFIX) $(PYTHON_RUN) -m pytest $(PYTEST_ARGS) tests/unit -m "unit and not requires_dat_file"

test-integration:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	$(PYTEST_ENV_PREFIX) $(PYTHON_RUN) -m pytest $(PYTEST_ARGS) tests/integration -m "integration and not requires_dat_file"

test-api:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	$(PYTEST_ENV_PREFIX) $(PYTHON_RUN) -m pytest $(PYTEST_ARGS) tests/api -m "api and not requires_dat_file"

test-all:
	@echo [make] Using PYTHON=$(PYTHON) [$(PYTHON_SOURCE)]
	$(PYTEST_ENV_PREFIX) $(PYTHON_RUN) -m pytest $(PYTEST_ARGS)
