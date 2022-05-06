import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class HMIDataset(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            source_max_token_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text_encoding = self.tokenizer(
            data_row["source_text"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return dict(
            input_ids=source_text_encoding["input_ids"].flatten(),
            attention_mask=source_text_encoding["attention_mask"].flatten(),
            efficacy=data_row.efficacy.flatten(),
            source=data_row.source_text,
            retrieval_score=data_row.score
        )