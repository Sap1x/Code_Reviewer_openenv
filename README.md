# 🚀 CodeRL — Agentic Code Review RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.org)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://docker.com)

> 🎯 **A benchmark system to evaluate AI code reviewers**

CodeRL is a production-grade, OpenEnv-compliant Reinforcement Learning environment that simulates real-world code review workflows. AI agents analyze pull request diffs, detect bugs and security vulnerabilities, and receive dense rewards based on precision, recall, and severity matching.

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
cd coderl
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_token_here"

python inference.py
```

---

## 🏗️ Architecture

```
coderl/
│
├── env/                    # Core RL environment
│   ├── environment.py      # Main CodeReviewEnv (reset/step/state)
│   ├── state.py            # Pydantic models (Observation, Action, etc.)
│   ├── reward.py           # Dense reward calculator
│   ├── grader.py           # Deterministic grader
│   ├── task_loader.py      # Task loading & validation
│   └── tools.py            # Simulated dev tools
│
├── data/                   # Code review tasks
│   ├── easy.json           # Syntax errors, simple bugs
│   ├── medium.json         # Logic errors, threading bugs
│   └── hard.json           # Security vulns, injection attacks
│
├── inference.py            # LLM-based baseline agent
├── server.py               # FastAPI HTTP server
├── openenv.yaml            # OpenEnv specification
├── Dockerfile              # Docker deployment
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

---

## 🎯 How It Works

### Environment Loop

```
Agent                     Environment
  │                           │
  │──── observe ──────────────│
  │                           │── load task, build observation
  │◄──── observation ─────────│
  │                           │
  │──── action (comments) ───►│
  │                           │── grade, calculate reward
  │◄──── reward + obs ────────│
  │                           │
  │  ... repeat for N steps   │
  │                           │
  │◄──── final score ─────────│
```

### OpenEnv Interface

| Method | Description |
|--------|-------------|
| `reset(task_id?)` | Start a new episode, returns initial observation |
| `step(action)` | Submit review comments, returns (obs, reward, done, info) |
| `state()` | Get current internal state |

### Task Difficulties

| Level | Issues | Examples |
|-------|--------|----------|
| 🟢 Easy | 4-7 | Off-by-one errors, resource leaks, missing validation |
| 🟡 Medium | 6-9 | Wrong business logic, threading bugs, API breakage |
| 🔴 Hard | 10-11 | SQL injection, command injection, path traversal, eval() |

---

## 🧮 Reward System

```python
reward = (
    precision * 0.5      # Are your findings correct?
    + recall * 0.3       # Did you find all the issues?
    + severity_bonus * 0.2  # Did you find the critical ones?
    - false_positive_penalty  # -0.5 per false positive
    - duplicate_penalty       # -0.2 per duplicate
)
```

| Scenario | Reward |
|----------|--------|
| Correct issue found | +1.0 |
| Partial match | +0.5 |
| Critical bug found | +bonus |
| False positive | -0.5 |
| Duplicate finding | -0.2 |

---

## 🛠️ Advanced Features

### Tool Simulation

Agents can invoke simulated developer tools during review:

```json
{
  "tool_calls": [
    {"tool": "inspect_function", "argument": "hash_password"},
    {"tool": "trace_variable", "argument": "username"}
  ]
}
```

### Multi-Step Reasoning

- **Step 1**: Identify obvious issues (syntax, missing checks)
- **Step 2**: Deeper inspection (logic errors, security)
- **Step 3**: Final review (subtle bugs, cross-function issues)

History is maintained across steps, enabling iterative refinement.

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t coderl .

# Run
docker run -p 7860:7860 coderl
```

### Hugging Face Spaces

The server responds to:
- `POST /reset` → HTTP 200 with initial observation
- `POST /step` → HTTP 200 with step result
- `GET /health` → HTTP 200 with status
- `GET /tasks` → HTTP 200 with task list

---

## 📡 API Reference

### POST /reset

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard_001"}'
```

### POST /step

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "comments": [
      {
        "line": 31,
        "issue": "SQL Injection",
        "severity": "critical",
        "explanation": "User input interpolated into SQL query",
        "suggestion": "Use parameterized queries",
        "confidence": 0.95
      }
    ]
  }'
```

---

## 🔥 Logging Format

Inference produces strict logging:

```
[START] task=hard_001 env=CodeRL model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=comments=3,tools=1 reward=0.4500 done=false error=null
[STEP] step=2 action=comments=2,tools=0 reward=0.3200 done=false error=null
[STEP] step=3 action=comments=1,tools=0 reward=0.1800 done=true error=null
[END] success=true steps=3 score=0.8234 rewards=[0.45, 0.32, 0.18]
```

---

## ⚙️ Performance

- **Runtime**: < 20 minutes for all tasks
- **Resources**: 2 vCPU, 8GB RAM
- **Deterministic**: Same inputs → same outputs

---

## 🏆 Design Philosophy

This is **not a toy environment**. It's a benchmark system where:
- ❌ Weak models fail to detect deep security vulnerabilities
- ✅ Strong models identify subtle cross-function bugs
- 📊 Results are measurable, reproducible, and comparable

Built like a product, not a project. 🚀
