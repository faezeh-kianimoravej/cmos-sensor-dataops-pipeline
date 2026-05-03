# OBSeRVeD CMOS VOC Sensor - MLOps Pipeline

End-to-end MLOps project for CMOS VOC sensor data.

Current workflow includes:

- ingestion from raw sensor files
- preprocessing and feature engineering
- stage-1 mixture detection model training
- stage-2 single-gas probability model training
- evaluation and batch prediction
- FastAPI inference endpoints

## Prerequisites

- Python 3.12 recommended on Windows, or Python 3.11+ if your dependencies support it
- Docker Desktop (running)
- Git
- GNU Make

### Install GNU Make

**Windows PowerShell must be run as Administrator.** Right-click PowerShell and select "Run as Administrator" before proceeding.

```powershell
# Using Chocolatey (recommended if you have it):
choco install make

# OR using winget (built-in on Windows 11):
winget install --id GnuWin32.Make -e
```

If Chocolatey is not installed, download and install it from https://chocolatey.org/install (also requires Administrator PowerShell).

**After install completes:** Close PowerShell completely (all windows). Then open a **new PowerShell window **.

Linux/macOS (usually pre-installed):

```bash
brew install make  # macOS
sudo apt-get install make  # Ubuntu/Debian
```

## Quick Start

### Fast Path

Shortest path for project evaluation:

1. Create the local Python environment and install dependencies.
2. Copy the example environment file to `.env`.
3. Start the platform services.
4. Run the pipeline.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
make platform-up
make pipeline-run
```

## Detailed Steps

If you want the full setup with explanations, follow the steps below.

### Step 0: Create environment configuration

Before running any commands, create `.env` from `.env.example`:

Windows (PowerShell):

```powershell
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

### Step 1: Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you see a venv copy error or a message telling you to run `python -m pip install --upgrade pip`, delete the `.venv` folder, recreate the virtual environment with Python 3.12, activate it again, and rerun the install step.

### Step 3: Run the full pipeline

```bash
make pipeline-run
```

Default raw input path in [configs/default.yaml](configs/default.yaml):

`data/raw/CMOS_addtional_datasets`

Direct fallback command:

```bash
python scripts/run_pipeline.py run --config configs/default.yaml --input data/raw/CMOS_addtional_datasets
```

## Common Commands

Run `make help` to list all commands.

Pipeline:

```bash
make pipeline-run
make pipeline-train
make pipeline-eval
make pipeline-predict
make clean-artifacts
```

Services:

```bash
make platform-up
make platform-down
make storage-up
make api-run
make airflow-run
make mlflow-run
```

Windows batch shortcut:

```powershell
scripts\run_pipeline.bat pipeline-run
scripts\run_pipeline.bat pipeline-train
scripts\run_pipeline.bat pipeline-eval
scripts\run_pipeline.bat pipeline-predict
scripts\run_pipeline.bat clean
```

## Testing

Recommended:

```bash
make test
make test-unit
make test-integration
make test-api
make test-all
```

Direct pytest fallback:

```bash
python -m pytest -q
python -m pytest -q tests/unit -m "unit and not requires_dat_file"
python -m pytest -q tests/integration -m "integration and not requires_dat_file"
python -m pytest -q tests/api -m "api and not requires_dat_file"
```

Notes:

- `make test` excludes `requires_dat_file`.
- `make test-all` includes all tests.
- On Windows, Make test targets set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for stable startup.

## API

The API exposes hierarchical inference for the two-stage pipeline.

Run locally:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /model-info?model_family=stage1|stage2`
- `POST /predict`
- `POST /predict-raw`

Swagger UI:

- `http://localhost:8000/docs`

`GET /health` returns the service status, uptime, version, and available model bundles.

`GET /model-info` returns metadata for either `stage1` or `stage2`.

`POST /predict` accepts a feature dictionary and runs the hierarchical stage-1/stage-2 inference path.

`POST /predict-raw` accepts raw time-series samples as `[second, sensor_value]` pairs, extracts features, and then runs the same inference path.

`POST /predict` request body supports:

- `run_id` (optional)
- `event_id` (optional)
- `features` (map of numeric feature names to values)

`POST /predict-raw` request body supports:

- `run_id` (optional)
- `event_id` (optional)
- `samples` (list of at least 10 `[second, sensor_value]` pairs)

### Raw Payload Test Fixtures

You can send these payloads directly to `POST /predict-raw` by copy/paste (Swagger UI, hosted API, or local API).

Single-gas sample payload:

```json
{
	"run_id": "synthetic_single_gas_test",
	"event_id": "single_gas_synthetic_1",
	"samples": [
		[0, 0.1],
		[1, 0.12],
		[2, 0.13],
		[3, 0.15],
		[4, 0.16],
		[5, 0.18],
		[6, 0.19],
		[7, 0.21],
		[8, 0.22],
		[9, 0.23],
		[10, 0.25],
		[11, 0.26],
		[12, 0.27],
		[13, 0.29],
		[14, 0.3],
		[15, 0.31],
		[16, 0.32],
		[17, 0.33],
		[18, 0.34],
		[19, 0.35],
		[20, 0.36],
		[21, 0.37],
		[22, 0.38],
		[23, 0.39],
		[24, 0.4],
		[25, 0.41],
		[26, 0.42],
		[27, 0.43],
		[28, 0.44],
		[29, 0.45],
		[30, 0.46],
		[31, 0.47],
		[32, 0.48],
		[33, 0.49],
		[34, 0.5],
		[35, 0.51],
		[36, 0.52],
		[37, 0.53],
		[38, 0.54],
		[39, 0.55],
		[40, 0.56],
		[41, 0.57],
		[42, 0.58],
		[43, 0.59],
		[44, 0.6],
		[45, 0.61],
		[46, 0.62],
		[47, 0.63],
		[48, 0.64],
		[49, 0.65]
	]
}
```

Mixture sample payload:

```json
{
	"run_id": "Mixing LowHigh concentrastions__UV ink-exposed to Toluene and 2 butanone",
	"event_id": "mixture-window-0-900",
	"samples": [
		[1100.901, 1.56283],
		[1102.091, 1.56041],
		[1103.374, 1.56908],
		[1104.548, 1.59832],
		[1105.823, 1.5959],
		[1106.96, 1.62242],
		[1108.138, 1.60345],
		[1109.356, 1.55934],
		[1110.527, 1.56277],
		[1111.808, 1.54221],
		[1112.974, 1.53423],
		[1114.124, 1.48939],
		[1115.353, 1.47566],
		[1116.474, 1.44473],
		[1117.606, 1.42989],
		[1118.901, 1.43556],
		[1120.041, 1.47036],
		[1121.34, 1.48005],
		[1122.515, 1.4614],
		[1123.811, 1.45935],
		[1124.977, 1.44139],
		[1126.273, 1.46141],
		[1127.443, 1.50474],
		[1128.737, 1.50496],
		[1129.882, 1.50525],
		[1131.168, 1.48389],
		[1132.367, 1.50279],
		[1133.619, 1.49756],
		[1134.786, 1.48553],
		[1136.112, 1.49565],
		[1137.288, 1.46765],
		[1138.543, 1.48188],
		[1139.74, 1.46667],
		[1141.015, 1.46559],
		[1142.181, 1.47568],
		[1143.465, 1.48388],
		[1144.63, 1.47073],
		[1145.93, 1.45133],
		[1147.129, 1.44808],
		[1148.379, 1.44786],
		[1149.767, 1.47471],
		[1150.974, 1.44837],
		[1152.228, 1.45759],
		[1153.389, 1.4491],
		[1154.677, 1.44068],
		[1155.829, 1.45666],
		[1156.98, 1.42905],
		[1158.197, 1.45019],
		[1159.603, 1.47231],
		[1160.758, 1.46581],
		[1162.043, 1.46357],
		[1163.227, 1.48966],
		[1164.498, 1.50578],
		[1165.667, 1.50643],
		[1166.96, 1.52222],
		[1168.176, 1.51966],
		[1169.492, 1.54705],
		[1170.82, 1.53448],
		[1172.237, 1.55425],
		[1173.588, 1.53767],
		[1174.959, 1.56632],
		[1176.189, 1.56463],
		[1177.457, 1.59115],
		[1178.785, 1.58096],
		[1180.202, 1.59705],
		[1181.563, 1.6252],
		[1182.718, 1.62639],
		[1184.003, 1.62144],
		[1185.221, 1.64721],
		[1186.582, 1.64548],
		[1187.849, 1.64862],
		[1189.292, 1.6397],
		[1190.614, 1.67667],
		[1192.009, 1.69817],
		[1193.208, 1.68372],
		[1194.564, 1.62078],
		[1195.854, 1.56737],
		[1197.259, 1.44362],
		[1198.619, 1.38906],
		[1199.795, 1.29016],
		[1201.08, 1.15188],
		[1202.254, 1.04847],
		[1203.522, 0.95127],
		[1204.721, 0.84233],
		[1205.975, 0.74018],
		[1207.128, 0.6384],
		[1208.455, 0.56255],
		[1209.659, 0.43677],
		[1210.909, 0.33712],
		[1212.109, 0.28621],
		[1213.341, 0.22702],
		[1214.54, 0.17958],
		[1215.829, 0.1811],
		[1217.008, 0.13894],
		[1218.275, 0.14219],
		[1219.637, 0.15625],
		[1220.854, 0.16015],
		[1222.105, 0.14458],
		[1223.617, 0.15944],
		[1224.905, 0.14997]
	]
}
```

PowerShell request example (works for local or hosted API by changing `API_BASE_URL`):

```powershell
$API_BASE_URL = "http://localhost:8000"
$payload = @'
{
	"run_id": "synthetic_single_gas_test",
	"event_id": "single_gas_synthetic_1",
	"samples": [[0,0.1],[1,0.12],[2,0.13],[3,0.15],[4,0.16],[5,0.18],[6,0.19],[7,0.21],[8,0.22],[9,0.23],[10,0.25],[11,0.26],[12,0.27],[13,0.29],[14,0.3],[15,0.31],[16,0.32],[17,0.33],[18,0.34],[19,0.35],[20,0.36],[21,0.37],[22,0.38],[23,0.39],[24,0.4],[25,0.41],[26,0.42],[27,0.43],[28,0.44],[29,0.45],[30,0.46],[31,0.47],[32,0.48],[33,0.49],[34,0.5],[35,0.51],[36,0.52],[37,0.53],[38,0.54],[39,0.55],[40,0.56],[41,0.57],[42,0.58],[43,0.59],[44,0.6],[45,0.61],[46,0.62],[47,0.63],[48,0.64],[49,0.65]]
}
'@

Invoke-RestMethod -Uri "$API_BASE_URL/predict-raw" -Method Post -ContentType "application/json" -Body $payload
```

`POST /predict` response highlights:

- `stage1_result.predicted_class`: `single_gas` or `mixture`
- `stage1_result.class_probabilities.single_gas`
- `stage1_result.class_probabilities.mixture`
- `stage2_result` is only present when stage 1 predicts `single_gas`
- `model_versions` returns validated loaded model names

Example response shape:

```json
{
	"run_id": "demo-run-001",
	"event_id": "evt-001",
	"route": "stage1_only",
	"stage1_result": {
		"predicted_class": "mixture",
		"class_probabilities": {
			"single_gas": 0.01,
			"mixture": 0.99
		}
	},
	"model_versions": {
		"stage1": "<resolved stage-1 model name>",
		"stage2": "<resolved stage-2 model name>"
	}
}
```

## Docker

Start all services:

```bash
docker-compose -f docker/docker-compose.yml up -d
```

Check status:

```bash
docker-compose -f docker/docker-compose.yml ps
```

Build API image (single service, ECS-ready base):

```bash
docker build -f docker/Dockerfile -t observed-api:local .
```

Run API container locally:

```bash
docker run --rm -p 8000:8000 observed-api:local
```

Optional: mount local data/models during local tests:

```bash
docker run --rm -p 8000:8000 -v ${PWD}/data:/app/data -v ${PWD}/artifacts:/app/artifacts observed-api:local
```

Verify container is healthy:

```bash
curl http://localhost:8000/health
```

PowerShell alternative:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Default ports:

- Airflow UI: `http://localhost:8080`
- MLflow UI: `http://localhost:5000`
- API: `http://localhost:8000`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

For storage, deployment, and infrastructure details, refer to:

- [docs/OBSeRVeD_MLOps_CMOS_VOC_Pipeline_FinalReport_v2.pdf](docs/OBSeRVeD_MLOps_CMOS_VOC_Pipeline_FinalReport_v2.pdf)
- [docs/plan_of_approach.md](docs/plan_of_approach.md)

## Key Output Paths

Pipeline artifacts:

- `data/processed/processed_timeseries.parquet`
- `data/processed/windowed_timeseries.parquet`
- `data/processed/features_timeseries.parquet`
- `data/processed/features_single_gas.parquet`
- `artifacts/models/stage1_mixture_detector/`
- `artifacts/models/stage2_gas_probability_model/`
- `artifacts/predictions/stage1_training_predictions.parquet`
- `artifacts/predictions/stage2_training_predictions.parquet`
- `artifacts/predictions/batch_predictions.parquet`
- `artifacts/reports/`

MLflow:

- `artifacts/mlflow/mlflow.db`
- `artifacts/mlflow/mlflow_tracking.db`
- `artifacts/mlruns/`

## Notes

- Run commands from repository root.
- Main runtime configuration: [configs/default.yaml](configs/default.yaml).
- Main orchestration wrapper: [scripts/run_pipeline.py](scripts/run_pipeline.py).

## CI/CD (GitLab)

Pipeline file: [.gitlab-ci.yml](.gitlab-ci.yml)

For CI/CD details, see the final report:

- [docs/OBSeRVeD_MLOps_CMOS_VOC_Pipeline_FinalReport_v2.pdf](docs/OBSeRVeD_MLOps_CMOS_VOC_Pipeline_FinalReport_v2.pdf)

## Documentation

- [docs/plan_of_approach.md](docs/plan_of_approach.md)
- [docs/team_charter.md](docs/team_charter.md)
- [docs/OBSeRVeD_MLOps_CMOS_VOC_Pipeline_FinalReport_v2.pdf](docs/OBSeRVeD_MLOps_CMOS_VOC_Pipeline_FinalReport_v2.pdf)
- [docs/LiteratureReview.pdf](docs/LiteratureReview.pdf)
