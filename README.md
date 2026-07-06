# judge-evals

Open-source **LLM-as-judge** evaluation harness: rubric-based scoring both **offline**
(batch evaluation, agreement vs. human labels) and as a **runtime gate** that judges an
agent/LLM output live and triggers retries on failure.

Judges run through [litellm](https://github.com/BerriAI/litellm), so any backend works —
a hosted API model or a local vLLM / ollama endpoint.

> Clean-room project. No proprietary code, prompts, or data.

## Architecture

```mermaid
flowchart LR
    subgraph inputs[Inputs]
        S[Sample<br/>input / output / context / reference]
        R[Rubric<br/>list of Criterion + scales]
    end

    S --> PB[Prompt builder<br/>rubric to JSON-schema prompt]
    R --> PB
    PB --> J{{Judge model<br/>via litellm}}
    J --> V[JudgeVerdict<br/>per-criterion score + rationale + pass/fail]

    subgraph offline[Offline batch runner]
        direction LR
        C[(sqlite cache<br/>sample x rubric x model x prompt-ver)]
        BC[Bias controls<br/>pairwise flip + self-consistency]
    end
    J <--> C
    V --> BC

    subgraph runtime[Runtime gate]
        G[JudgeGate.gate fn, rubric, policy]
        P[Policy<br/>retry w/ feedback / escalate / reject]
    end
    V --> G
    G --> P
    P -.retry.-> J
```

## Quickstart

```bash
make setup     # uv sync --dev  (creates .venv, installs deps)
make test      # run the pytest suite
make lint      # ruff check + format --check
make run       # invoke the judge-evals CLI
```

Requires [uv](https://docs.astral.sh/uv/). Python 3.11+ is provisioned by uv itself.

Copy `.env.example` to `.env` and fill in provider keys for whichever judge backend you use.

