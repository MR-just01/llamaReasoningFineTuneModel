# Llama 3.2 Reasoning API

A FastAPI-based inference service for serving a fine-tuned **Llama 3.2 3B Instruct** model through REST API endpoints.

The application loads a pretrained Llama model, attaches a fine-tuned LoRA adapter from Hugging Face, and provides an API for generating responses to reasoning-oriented questions.

## Project Overview

This repository focuses on the model-serving and inference layer of the Llama 3.2 Reasoning project.

Instead of loading the model for every request, the application initializes it during application startup and reuses the loaded model for subsequent inference requests.

The API provides:

- Text generation through a REST endpoint.
- Request validation using Pydantic.
- Model initialization and lifecycle management.
- Health and readiness endpoints.
- Structured error handling.
- Request latency and token-count tracking.
- Evaluation and API benchmarking scripts.
- Docker-based deployment configuration.

## Architecture

```text
             Client
               |
               v
         FastAPI Application
               |
               v
          API Routes
               |
               v
       Request Validation
            (Pydantic)
               |
               v
       Generation Service
               |
               v
        Inference Engine
               |
        +------+------+
        |             |
        v             v
   Base Llama     LoRA Adapter
    3.2 3B        Hugging Face
        |             |
        +------+------+
               |
               v
       Generated Response
               |
               v
         JSON Response
```

## Model Configuration

The application uses the following models:

| Component | Model |
|---|---|
| Base Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Fine-Tuned Adapter | `MR023/Llama3.2-Reasoning` |

The adapter is loaded onto the base model using PEFT.

The model is initialized once during application startup through FastAPI's lifespan management.

## Technology Stack

- **Python** — Core programming language
- **FastAPI** — REST API framework
- **Uvicorn** — ASGI server
- **PyTorch** — Model execution
- **Hugging Face Transformers** — Model and tokenizer loading
- **PEFT** — LoRA adapter integration
- **Pydantic** — Request and response validation
- **Docker** — Containerization

## Project Structure

```text
llamaReasoningFineTuneModel/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   └── routes.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── exceptions.py
│   │
│   ├── models/
│   │   ├── loader.py
│   │   └── inference.py
│   │
│   ├── schemas/
│   │   └── request.py
│   │
│   └── services/
│       └── generation_service.py
│
├── scripts/
│   ├── evaluate_model.py
│   ├── api_benchmark.py
│   └── dataset_api_benchmark.py
│
├── tests/
│   └── test_api.py
│
├── data/
├── results/
├── Dockerfile
├── requirements.txt
└── README.md
```

## API Endpoints

### 1. Generate Response

**Endpoint:** `POST /generate`

Generates a response using the fine-tuned reasoning model.

Example request:

```json
{
  "instruction": "Solve the following problem step by step.",
  "input": "If 5 workers complete a task in 10 days, how many days will 10 workers take?",
  "max_new_tokens": 512
}
```

Example response structure:

```json
{
  "text": "The total work is 5 × 10 = 50 worker-days...",
  "input_tokens": 35,
  "output_tokens": 85,
  "total_tokens": 120
}
```

The generated text and token counts depend on the actual model response.

### 2. Health Check

**Endpoint:** `GET /health`

Checks whether the API application is running.

### 3. Readiness Check

**Endpoint:** `GET /ready`

Checks whether the model is ready to handle inference requests.

## Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- Sufficient RAM and GPU memory for the selected model configuration.
- Hugging Face access to the base model.
- NVIDIA GPU and compatible CUDA environment recommended.

### 1. Clone the Repository

```bash
git clone https://github.com/MR-just01/llamaReasoningFineTuneModel.git

cd llamaReasoningFineTuneModel
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Install a compatible PyTorch build separately if it is not already available in your environment. The appropriate build depends on your operating system and CUDA setup.

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
MODEL_ID=meta-llama/Llama-3.2-3B-Instruct
ADAPTER_ID=MR023/Llama3.2-Reasoning

HF_TOKEN=your_huggingface_token

MAX_NEW_TOKENS=512
TORCH_DTYPE=float16
DEVICE_MAP=auto

HOST=0.0.0.0
PORT=8000
```

Use your own Hugging Face access token. Never commit the `.env` file or expose your token publicly.

### 5. Start the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available locally at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

## Docker Deployment

The repository includes a Dockerfile configured with an NVIDIA CUDA runtime.

Build the Docker image:

```bash
docker build -t llama-reasoning-api .
```

Run the container:

```bash
docker run --gpus all \
  -p 8000:8000 \
  --env-file .env \
  llama-reasoning-api
```

The container requires a compatible NVIDIA driver and NVIDIA Container Toolkit.

**Note:** Docker configuration makes the application container-ready. It does not, by itself, mean that the API is hosted or continuously available online.

## Evaluation and Benchmarking

The repository includes scripts for evaluating the API and observing its inference performance.

| Script | Purpose |
|---|---|
| `evaluate_model.py` | Evaluates generated responses against an audited dataset. |
| `api_benchmark.py` | Measures API request latency and related performance statistics. |
| `dataset_api_benchmark.py` | Runs API inference across a dataset and records latency, token counts, and throughput. |

The benchmarking scripts provide an initial performance assessment. Larger and repeated test runs are needed for reliable production latency percentiles.

## Error Handling and Validation

The application uses Pydantic schemas to validate incoming requests.

The generation service also handles inference-related failures and provides structured error responses.

The API includes checks for model availability before processing generation requests.

## Related Repository

The model training and fine-tuning experiments are maintained separately:

[**Llama3.2-Reasoning — Model Development Repository**](https://github.com/MR-just01/Llama3.2-Reasoning)

The Hugging Face adapter used by this API:

[**MR023/Llama3.2-Reasoning**](https://huggingface.co/MR023/Llama3.2-Reasoning)

## Future Improvements

- Add end-to-end API integration tests.
- Improve concurrent inference handling.
- Add authentication and request rate limiting.
- Introduce structured monitoring and logging.
- Run larger-scale latency and throughput benchmarks.
- Add deployment-specific configuration and operational documentation.

## Project Focus

This repository focuses on **serving a fine-tuned language model through a modular REST API**.

It separates model loading, inference, API routing, request validation, and generation services to make the application easier to maintain and extend.
