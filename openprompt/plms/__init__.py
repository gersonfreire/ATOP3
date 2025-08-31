from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from .utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from .mlm import MLMTokenizerWrapper
from .seq2seq import T5TokenizerWrapper
from .lm import LMTokenizerWrapper
from transformers import BertConfig, BertTokenizer, BertModel,BertForSequenceClassification, BertForMaskedLM, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, \
                         AlbertTokenizer, AlbertConfig, AlbertModel, AlbertForMaskedLM, \
                         T5Config, T5Tokenizer, T5ForConditionalGeneration, \
                         OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, \
                         GPT2Config, GPT2Tokenizer, GPT2LMHeadModel      
from collections import namedtuple
from yacs.config import CfgNode
from transformers import AutoTokenizer
from transformers import AutoModel

from openprompt.utils.logging import logger
from transformers import logging
logging.set_verbosity_error()

    
ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model'))
# Define as classes de modelo para tarefas downstream
_MODEL_CLASSES = {
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model': BertForMaskedLM,
    }),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaForMaskedLM,
    }),
    'albert': ModelClass(**{
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model':AlbertForMaskedLM,
    }),
    'gpt': ModelClass(**{
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTLMHeadModel,
    }),
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel,
    }),
    't5':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
    }),
}

TOKENIZER_WRAPPER_MAPPING = {
    BertTokenizer: MLMTokenizerWrapper,
    RobertaTokenizer: MLMTokenizerWrapper,
    AlbertTokenizer: MLMTokenizerWrapper,
    OpenAIGPTTokenizer: LMTokenizerWrapper,
    GPT2Tokenizer: LMTokenizerWrapper,
    T5Tokenizer: T5TokenizerWrapper,
}

def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]

def get_tokenizer_wrapper(tokenizer: PreTrainedTokenizer) -> TokenizerWrapper:
    try:
        wrapper_class = TOKENIZER_WRAPPER_MAPPING[type(tokenizer)]
    except KeyError:
        logger.info("tokenizer type not in TOKENIZER_WRAPPER_MAPPING")
    return wrapper_class


def load_plm(config: CfgNode):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.
    
    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
    
    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
    """
    plm_config = config.plm
    model_class = get_model_class(plm_type = plm_config.model_name)
    model_config = model_class.config.from_pretrained(plm_config.model_path)
    model_config.output_hidden_states = True
    # você pode alterar o model_config do huggingface aqui
    model = model_class.model.from_pretrained(plm_config.model_path, config=model_config,ignore_mismatched_sizes=True)
    # Congela os parâmetros do modelo
    for name, param in model.named_parameters():
        param.requires_grad = False
    tokenizer = model_class.tokenizer.from_pretrained(plm_config.model_path)

    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=config.plm.specials_to_add)
    return model, tokenizer, model_config

def add_special_tokens(model: PreTrainedModel, 
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    r"""Adiciona os tokens especiais ao tokenizer se o token especial
    não estiver presente no tokenizer.

    Args:
        model (:obj:`PreTrainedModel`): O modelo pré-treinado para redimensionar o embedding
                após adicionar tokens especiais.
        tokenizer (:obj:`PreTrainedTokenizer`): O tokenizer pré-treinado para adicionar tokens especiais.
        specials_to_add: (:obj:`List[str]`, opcional): Os tokens especiais a serem adicionados. Padrão é o token de preenchimento (pad).

    Returns:
        O modelo redimensionado, o tokenizer com os tokens especiais adicionados.

    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                logger.info("pad token é None, definido para o id {}".format(tokenizer.pad_token_id))
    return model, tokenizer



