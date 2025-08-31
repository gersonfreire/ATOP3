# Copilot Instructions for ATOP3

## Project Overview
- Implements "ATOP: Adversarial Topic-aware Prompt-tuning for Cross-topic Automated Essay Scoring".
- Built on top of [OpenPrompt](https://github.com/thunlp/OpenPrompt); see `openprompt/` for core logic and extensions.
- Main experiment entrypoint: `experiments/cli.py` (uses YAML config, e.g., `soft_prompt.yaml`).

## Directory Structure
- `datasets/AES/`: Essay scoring datasets, split scripts, and preprocessed `.pk` files.
- `openprompt/`: Core prompt-tuning framework, including trainers, pipelines, data utils, and PLM wrappers.
- `experiments/`: Experiment scripts, configs, and shell helpers.
- `logs/`: Output logs from training and experiments.
- `yaml/`: Configuration files (e.g., `config.yaml`).

## Developer Workflows
- **Setup:**
  - Python 3.7+ required. Install dependencies with `pip install -r requirements.txt`.
  - Use a virtual environment for isolation.
- **Run Experiments:**
  - Standard: `python experiments/cli.py --config_yaml soft_prompt.yaml`
  - With logging: `PYTHONUNBUFFERED=1 python -u experiments/cli.py --config_yaml soft_prompt.yaml 2>&1 | tee -a logs/train_$(date +%F_%H-%M-%S).log`
- **Dataset Generation:**
  - Use `datasets/AES/generate_pk_from_csv.py` and `split_class.py` to preprocess and split data.
- **Configurable Training:**
  - Modify YAML files in `experiments/` or `yaml/` to change experiment parameters.

## Project-Specific Patterns
- **Prompt-tuning logic** is modularized in `openprompt/prompt_base.py` and `openprompt/lm_bff_trainer.py`.
- **Data loading** uses custom dataset classes in `openprompt/data_utils/`.
- **Trainer and pipeline** patterns follow OpenPrompt conventions but may be extended for adversarial/topic-aware logic.
- **Logging** is handled via shell redirection and log files in `logs/`.
- **Error Handling:**
  - Common errors: missing modules (`datasets.AES.split_class`, `openprompt.prompts`), version mismatches (e.g., `AdamW` import).
  - Check `requirements.txt` and ensure correct Python version.

## Integration Points
- Relies on HuggingFace Transformers and OpenPrompt for model and optimization logic.
- Data flows: CSV → PK (pickle) via scripts in `datasets/AES/`, then loaded by OpenPrompt modules.
- Config-driven: Most experiment parameters are set via YAML files.

## Example: Adding a New Experiment
1. Create a new YAML config in `experiments/` or `yaml/`.
2. Run: `python experiments/cli.py --config_yaml <your_config>.yaml`
3. Logs will be saved in `logs/`.

## References
- See `README.md` for setup and run instructions.
- See `openprompt/` for framework extensions and custom logic.
- See `datasets/AES/` for data preparation scripts and formats.

---

If any section is unclear or missing important details, please provide feedback to improve these instructions.
