import abc
from copy import deepcopy
from typing import Optional, Mapping, Any, List
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Mapping, Union, Iterable, Optional, Tuple

from spacy.lang.en import English
from torch.nn import DataParallel
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
import torch
from spacy.lang.en import English
from transformers import T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer

TokenizerReturnType = Mapping[str, Union[torch.Tensor, List[int],
                                         List[List[int]],
                                         List[List[str]]]]
DecodedOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


@torch.no_grad()
def greedy_decode(model: PreTrainedModel,
                  input_ids: torch.Tensor,
                  length: int,
                  attention_mask: torch.Tensor = None,
                  return_last_logits: bool = True) -> DecodedOutput:
    decode_ids = torch.full((input_ids.size(0), 1),
                            model.config.decoder_start_token_id,
                            dtype=torch.long).to(input_ids.device)
    encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
    next_token_logits = None
    for _ in range(length):
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True)
        wrapped_model = DataParallel(model, device_ids=[0,1,2,3])
        outputs = wrapped_model(model(**model_inputs))  # (batch_size, cur_len, vocab_size)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
        decode_ids = torch.cat([decode_ids,
                                next_token_logits.max(1)[1].unsqueeze(-1)],
                               dim=-1)
    if return_last_logits:
        return decode_ids, next_token_logits
    return decode_ids
prediction_tokens = {
        'castorini/monot5-base-msmarco':             ['▁false', '▁true'],
        'castorini/monot5-base-msmarco-10k':         ['▁false', '▁true'],
        'castorini/monot5-large-msmarco':            ['▁false', '▁true'],
        'castorini/monot5-large-msmarco-10k':        ['▁false', '▁true'],
        'castorini/monot5-base-med-msmarco':         ['▁false', '▁true'],
        'castorini/monot5-3b-med-msmarco':           ['▁false', '▁true'],
        'castorini/monot5-3b-msmarco-10k':           ['▁false', '▁true'],
        'unicamp-dl/mt5-base-en-msmarco':            ['▁no'   , '▁yes'],
        'unicamp-dl/ptt5-base-pt-msmarco-10k-v2':    ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v2':   ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2':['▁não'  , '▁sim'],
        'unicamp-dl/mt5-base-en-pt-msmarco-v2':      ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-mmarco-v2':             ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-en-pt-msmarco-v1':      ['▁no'   , '▁yes'],
        'unicamp-dl/mt5-base-mmarco-v1':             ['▁no'   , '▁yes'],
        'unicamp-dl/ptt5-base-pt-msmarco-10k-v1':    ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-pt-msmarco-100k-v1':   ['▁não'  , '▁sim'],
        'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['▁não'  , '▁sim']
        }



class Query:
    """Class representing a query.
    A query contains the query text itself and potentially other metadata.
    Parameters
    ----------
    text : str
        The query text.
    id : Optional[str]
        The query id.
    """

    def __init__(self, text: str, id: Optional[str] = None):
        self.text = text
        self.id = id


class Text:
    """Class representing a text to be reranked.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.
    Parameters
    ----------
    text : str
        The text to be reranked.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the text. For example, the score might be the BM25 score
        from an initial retrieval stage.
    title : Optional[str]
        The text's title.
    """

    def __init__(self,
                 text: str,
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0,
                 title: Optional[str] = None):
        self.text = text
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.title = title



TextType = Union['Query', 'Text']
@dataclass
class TokenizerOutputBatch:
    output: TokenizerReturnType
    texts: List[TextType]

    def __len__(self):
        return len(self.texts)


@dataclass
class QueryDocumentBatch:
    query: Query
    documents: List[Text]
    output: Optional[TokenizerReturnType] = None

    def __len__(self):
        return len(self.documents)


@dataclass
class DuoQueryDocumentBatch:
    query: Query
    doc_pairs: List[Tuple[Text, Text]]
    output: Optional[TokenizerReturnType] = None

    def __len__(self):
        return len(self.doc_pairs)


class TokenizerEncodeMixin:
    tokenizer: PreTrainedTokenizer = None
    tokenizer_kwargs = None

    def encode(self, strings: List[str]) -> TokenizerReturnType:
        assert self.tokenizer and self.tokenizer_kwargs is not None, \
            'mixin used improperly'
        ret = self.tokenizer.batch_encode_plus(strings,
                                               **self.tokenizer_kwargs)
        ret['tokens'] = list(map(self.tokenizer.tokenize, strings))
        return ret


class BatchTokenizer(TokenizerEncodeMixin):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int,
                 **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer_kwargs = tokenizer_kwargs

    def traverse(
            self,
            batch_input: List[TextType]) -> Iterable[TokenizerOutputBatch]:
        for batch_idx in range(0, len(batch_input), self.batch_size):
            inputs = batch_input[batch_idx:batch_idx + self.batch_size]
            input_ids = self.encode([x.text for x in inputs])
            yield TokenizerOutputBatch(input_ids, inputs)


class QueryDocumentBatchTokenizer(TokenizerEncodeMixin):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int,
                 pattern: str = '{query} {document}',
                 **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer_kwargs = tokenizer_kwargs
        self.pattern = pattern

    def traverse_query_document(
            self,
            batch_input: QueryDocumentBatch) -> Iterable[QueryDocumentBatch]:
        query = batch_input.query
        for batch_idx in range(0, len(batch_input), self.batch_size):
            docs = batch_input.documents[batch_idx:batch_idx + self.batch_size]
            outputs = self.encode([self.pattern.format(
                query=query.text,
                document=doc.text) for doc in docs])
            yield QueryDocumentBatch(query, docs, outputs)

    def traverse_duo_query_document(
            self,
            batch_input: DuoQueryDocumentBatch) -> Iterable[DuoQueryDocumentBatch]:
        query = batch_input.query
        for batch_idx in range(0, len(batch_input), self.batch_size):
            docs = batch_input.doc_pairs[batch_idx:batch_idx + self.batch_size]
            outputs = self.encode([self.pattern.format(
                query=query.text,
                document0=doc[0].text,
                document1=doc[1].text) for doc in docs])
            yield DuoQueryDocumentBatch(query, docs, outputs)


class T5BatchTokenizer(QueryDocumentBatchTokenizer):
    def __init__(self, *args, **kwargs):
        kwargs['pattern'] = 'Query: {query} Document: {document} Relevant:'
        if 'return_attention_mask' not in kwargs:
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512
        super().__init__(*args, **kwargs)


class T5DuoBatchTokenizer(QueryDocumentBatchTokenizer):
    def __init__(self, *args, **kwargs):
        kwargs['pattern'] = 'Query: {query} Document0: {document0} Document1: {document1} Relevant:'
        if 'return_attention_mask' not in kwargs:
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512
        super().__init__(*args, **kwargs)


class SimpleBatchTokenizer(BatchTokenizer):
    def __init__(self, *args, **kwargs):
        if 'return_attention_mask' not in kwargs:
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        super().__init__(*args, **kwargs)


class SpacyWordTokenizer:
    nlp = English()
    tokenizer = nlp.tokenizer

    @lru_cache(maxsize=1024)
    def __call__(self, text: str) -> List[str]:
        return list(x.text for x in self.tokenizer(text))


class SpacySenticizer:
    nlp = English()
    nlp.add_pipe('sentencizer')

    def __init__(self, max_paragraph_length: int = None):
        self.max_paragraph_length = max_paragraph_length

    @lru_cache(maxsize=1024)
    def __call__(self, document: str) -> List[str]:
        return [s.text for s in self.nlp(
            document[:self.max_paragraph_length]).sents]




class Reranker:
    """Class representing a reranker.
    A reranker takes a list texts and returns a list of texts non-destructively
    (i.e., does not alter the original input list of texts).
    """

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        """Sorts a list of texts
        """
        return sorted(self.rescore(query, texts), key=lambda x: x.score, reverse=True)

    @abc.abstractmethod
    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        """Reranks a list of texts with respect to a query.
         Parameters
         ----------
         query : Query
             The query.
         texts : List[Text]
             The list of texts.
         Returns
         -------
         List[Text]
             Reranked list of texts.
         """
        pass


class MonoT5(Reranker):
    def __init__(self,
                 pretrained_model_name_or_path: str = 'castorini/monot5-base-msmarco',
                 model: T5ForConditionalGeneration = None,
                 tokenizer=None,
                 use_amp=False,
                 token_false=None,
                 token_true=None):
        self.model = model or self.get_model(pretrained_model_name_or_path)
        self.tokenizer = tokenizer or self.get_tokenizer(pretrained_model_name_or_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
            pretrained_model_name_or_path, self.tokenizer, token_false, token_true)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str,
                  *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path,
                                                     *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str,
                      *args, batch_size: int = 128, **kwargs) -> T5BatchTokenizer:
        return T5BatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs),
            batch_size=batch_size
        )
    @staticmethod
    def get_prediction_tokens(pretrained_model_name_or_path: str,
            tokenizer, token_false, token_true):
        if not (token_false and token_true):
            if pretrained_model_name_or_path in prediction_tokens:
                token_false, token_true = prediction_tokens[pretrained_model_name_or_path]
                token_false_id = tokenizer.tokenizer.get_vocab()[token_false]
                token_true_id  = tokenizer.tokenizer.get_vocab()[token_true]
                return token_false_id, token_true_id
            else:
                raise Exception("We don't know the indexes for the non-relevant/relevant tokens for\
                        the checkpoint {pretrained_model_name_or_path} and you did not provide any.")
        else:
            token_false_id = tokenizer.tokenizer.get_vocab()[token_false]
            token_true_id  = tokenizer.tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id


    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in tqdm(self.tokenizer.traverse_query_document(batch_input),total=int(len(batch_input)/ self.tokenizer.batch_size)):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output['input_ids'].to(self.device)
                attn_mask = batch.output['attention_mask'].to(self.device)
                _, batch_scores = greedy_decode(self.model,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score

        return texts