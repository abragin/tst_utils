import torch
from torch.utils.data import Dataset


class StyleTransferDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_length, # For source & target separately
        is_train: bool,
        model_type: str,  # "GPT" or "T5"
    ):
        assert model_type in {"GPT", "T5"}

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.model_type = model_type

        self.has_reverse = (
            is_train and "styled_text_style_emb" in df.columns
        )

    def __len__(self):
        if self.is_train and self.has_reverse:
            return 2 * len(self.df)
        return len(self.df)

    def _tokenize_gpt_pair(self, src, tgt):
        # tokenize without EOS
        src_enc = self.tokenizer(
            src,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        tgt_enc = self.tokenizer(
            tgt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        src_ids = src_enc["input_ids"]
        tgt_ids = tgt_enc["input_ids"]
        eos_id = self.tokenizer.eos_token_id

        input_ids = (
            src_ids
            + [eos_id]
            + tgt_ids
            + [eos_id]
        )

        attention_mask = [1] * len(input_ids)

        labels = (
            [-100] * (len(src_ids) + 1)
            + tgt_ids
            + [eos_id]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __getitem__(self, idx):
        if isinstance(idx, (list, tuple)):
            return [self[i] for i in idx]
        # determine direction
        if self.is_train and self.has_reverse:
            row_idx = idx // 2
            reverse = (idx % 2) == 1
        else:
            row_idx = idx
            reverse = False

        row = self.df.iloc[row_idx]

        if self.is_train and reverse:
            src_text = row["text"]
            tgt_text = row["styled_text"]
            style = row["styled_text_style_emb"]
        else:
            src_text = row["styled_text"]
            tgt_text = row["text"]
            style = row["text_style_emb"]

        if self.model_type == "GPT":
            enc = self._tokenize_gpt_pair(src_text, tgt_text)
            enc["style"] = torch.tensor(style, dtype=torch.float16)
            return enc

        # ---- T5 ----
        enc = self.tokenizer(
            src_text,
            text_target=tgt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        # squeeze batch dim
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["style"] = torch.tensor(style, dtype=torch.float16)
        return enc

class StyleTransferCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        # separate style
        styles = torch.stack([f.pop("style") for f in features])

        # lengths
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            l = len(f["input_ids"])
            pad_len = max_len - l

            input_ids.append(
                f["input_ids"] + [self.pad_id] * pad_len
            )
            attention_mask.append(
                f["attention_mask"] + [0] * pad_len
            )
            labels.append(
                f["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "style": styles,
        }