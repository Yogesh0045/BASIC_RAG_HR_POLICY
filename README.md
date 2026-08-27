# HR RAG Policy Agent

A Python 3.14 project for working with an HR policy retrieval-augmented generation (RAG) agent.

## Requirements

- Python 3.14 or later
- `uv`
- The existing virtual environment: `basicragenv`

## Windows Command Prompt

From the project directory:

```cmd
basicragenv\Scripts\activate.bat
python --version
uv pip install -r requirements.txt
```

The prompt should show `(basicragenv)` after activation.

## Bash

For Git Bash on Windows, run:

```bash
source basicragenv/Scripts/activate
python --version
uv pip install -r requirements.txt
```

For a Unix-like environment where the virtual environment was created there:

```bash
source basicragenv/bin/activate
python --version
uv pip install -r requirements.txt
```

The Python version should be 3.14 or later, and the prompt should show `(basicragenv)` after activation.

## Run the Project

After installing the project package into `basicragenv`, run the configured console script:

```bash
uv pip install -e .
uv run hr-rag-policy-agent
```

The current entry point prints a starter message while the agent is being developed.

## Project Layout

```text
data/                          HR policy source documents
src/hr_rag_policy_agent/       Application package
requirements.txt               Runtime dependencies
pyproject.toml                 Project metadata and build configuration
basicragenv/                   Local Python virtual environment
```
