# ATOP

Code for Paper "ATOP:Adversarial TOpic-aware Prompt-tuning for Cross-topic Automated Essay Scoring"

### Requirements

The project requires Python 3.7 or higher. All dependencies are listed in `requirements.txt`.

### Setup

To set up the environment, follow these steps:

1. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run

```
python experiments/cli.py --config_yaml experiments/soft_prompt.yaml
```

### Para gerar log

```
mkdir -p logs
PYTHONUNBUFFERED=1 python -u experiments/cli.py --config_yaml soft_prompt.yaml 2>&1 | tee -a logs/train_$(date +%F_%H-%M-%S).log
```

### Credits

```
The code is built based on the open-source toolkit [OpenPrompt](https://github.com/thunlp/OpenPrompt). 
```

`ImportError: cannot import name 'AdamW' from 'transformers.optimization'`

ModuleNotFoundError: No module named 'datasets.AES.split_class'.

ModuleNotFoundError: No module named 'openprompt.prompts'
