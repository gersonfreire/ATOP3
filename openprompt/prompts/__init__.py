import torch
import torch.nn as nn
from typing import Optional, Sequence

from openprompt.prompt_base import Template, Verbalizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modeling_utils import PreTrainedModel
from yacs.config import CfgNode


class BasicTemplate(Template):
    """
    Minimal template that places a mask token after the input text.
    Produces required flags: loss_ids, shortenable_ids, soft_token_ids, new_token_ids.
    """

    # include soft/new token flags expected by our tokenizer wrapper
    registered_inputflag_names = ["loss_ids", "shortenable_ids", "soft_token_ids", "new_token_ids"]

    def __init__(self, tokenizer: PreTrainedTokenizer, mask_token: str = "<mask>"):
        super().__init__(tokenizer=tokenizer, mask_token=mask_token)
        # default template: text then one mask
        self.text = ["<text_a>", self.mask_token]

    def get_default_soft_token_ids(self):
        # 0 means no soft tokens
        return [0 for _ in self.text]

    def get_default_new_token_ids(self):
        # 0 means not a new token id placeholder
        return [0 for _ in self.text]

    def process_batch(self, batch):
        """Pass-through for standard token-based models (no soft tokens)."""
        # batch is either an InputFeatures or a dict-like
        if hasattr(batch, "to_dict"):
            data = batch.to_dict()
        elif isinstance(batch, dict):
            data = batch
        else:
            # fallback to attribute dict
            data = batch.__dict__
        # No special processing needed; the underlying model will consume input_ids/attention_mask
        return data


class AESVerbalizer(Verbalizer):
    """
    Simple learnable projection from hidden states to label logits and AES attribute scores.
    Returns (label_words_logits, attributes_pre) where attributes_pre is shaped (B, 9) in [0,1].
    """

    def __init__(self, tokenizer: Optional[PreTrainedTokenizer], classes: Optional[Sequence[str]], hidden_size: int = 768):
        super().__init__(tokenizer=tokenizer, classes=classes, num_classes=len(classes) if classes is not None else None)
        self.classifier = nn.Linear(hidden_size, self.num_classes or 2)
        self.attr_head = nn.Linear(hidden_size, 9)  # score, content, organization, word_choice, sentence_fluency, conventions, prompt_adherence, language, narrativity

    def generate_parameters(self, **kwargs):
        # Parameters are registered as module params already
        return [p for p in self.parameters()]

    def process_outputs(self, outputs, batch):
        # outputs shape: (B, H) or (B, T, H); reduce if needed
        if outputs.dim() == 3:
            # average across masked positions
            outputs = outputs.mean(dim=1)
        logits = self.classifier(outputs)
        attributes = torch.sigmoid(self.attr_head(outputs))
        return logits, attributes


def load_template(config: CfgNode, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, plm_config=None, **kwargs) -> Template:
    # Minimal loader that always returns BasicTemplate; respects tokenizer and mask token
    return BasicTemplate(tokenizer=tokenizer)


def load_verbalizer(config: CfgNode, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, plm_config=None, classes=None, **kwargs) -> Verbalizer:
    hidden_size = getattr(model.config, "hidden_size", 768)
    return AESVerbalizer(tokenizer=tokenizer, classes=classes, hidden_size=hidden_size)


# Optional placeholders to satisfy imports
def load_template_generator(*args, **kwargs):
    raise NotImplementedError("Template generator is not implemented in this minimal setup.")


def load_verbalizer_generator(*args, **kwargs):
    raise NotImplementedError("Verbalizer generator is not implemented in this minimal setup.")
