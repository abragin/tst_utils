import torch
import numpy as np
from tqdm import tqdm

# --- Predefined generation option sets ---
GEN_OPTS_V1 = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.0,
}

GEN_OPTS_V2 = {   # beam search
    "num_beams": 4,
    "early_stopping": True,
}

GEN_OPTS_V3 = {   # default TinyStyler sampling params
    "do_sample": True,
    "top_p": 0.8,
    "temperature": 1.0,
}

class TSTGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        target_styles,
        model_type="T5",  # "T5" or "GPT"
        style_emb_dict=None,  # if None → use author tags
        batch_size=8,
        num_sequences=1,
        generate_options=GEN_OPTS_V1,
        max_length=512,
        min_length=32,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_styles = target_styles
        self.model_type = model_type.upper()
        self.style_emb_dict = style_emb_dict
        self.batch_size = batch_size
        self.num_sequences = num_sequences
        self.generate_options = generate_options
        self.max_length = max_length
        self.min_length = min_length

        self.device = next(model.parameters()).device

    def _get_style_representation(self, target_style, batch_size):
        """Return style embeddings or author tag string depending on setup."""
        if self.style_emb_dict is not None:  # TinyStyler with embeddings
            if target_style not in self.style_emb_dict:
                raise ValueError(f"Unsupported style: {target_style}")
            emb = self.style_emb_dict[target_style]
            emb = np.array([emb] * batch_size)
            return torch.from_numpy(emb).float().to(self.device)
        else:  # use author tags
            if target_style not in self.target_styles:
                raise ValueError(f"Unsupported style: {target_style}")
            return f"<{target_style.lower()}>"

    def perform_tst(self, texts, target_style):
        """
        Perform text style transfer on a list of input texts.
        Returns a list of generated strings.
        """
        results = []
        total = len(texts)

        for start in tqdm(range(0, total, self.batch_size)):
            end = start + self.batch_size
            batch_texts = texts[start:end]

            if self.style_emb_dict is not None:  # style embeddings path
                style_embs = self._get_style_representation(target_style, len(batch_texts))

                # Tokenize
                if self.model_type == "GPT":
                    ipt = self.tokenizer(
                        [t + self.tokenizer.eos_token for t in batch_texts],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=int(self.max_length * 0.6),
                    )
                else:  # T5
                    ipt = self.tokenizer(
                        list(batch_texts),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                    )

                ipt = {k: v.to(self.device) for k, v in ipt.items()}

                # Generate
                output_sequences = self.model.generate(
                    input_ids=ipt["input_ids"],
                    attention_mask=ipt["attention_mask"],
                    style=style_embs,
                    max_length=self.max_length,
                    min_length=self.min_length,
                    num_return_sequences=self.num_sequences,
                    **self.generate_options,
                )
                decoded = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                grouped = [decoded[i:i + self.num_sequences] for i in range(0, len(decoded), self.num_sequences)]
            else:  # author tags path
                style_token = self._get_style_representation(target_style, len(batch_texts))

                if self.model_type == "GPT":
                    inputs = [txt + " " + style_token for txt in batch_texts]
                else:  # T5
                    inputs = [style_token + " " + txt for txt in batch_texts]

                ipt = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)

                output_ids = self.model.generate(
                    **ipt,
                    max_length=self.max_length,
                    min_length=self.min_length,
                    num_return_sequences=self.num_sequences,
                    **self.generate_options,
                )

                # For GPT with tags, strip everything before/including tag if needed
                if self.model_type == "GPT":
                    decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
                    cleaned = []
                    for t in decoded:
                        if style_token in t:
                            t = t.split(style_token, 1)[1].split(self.tokenizer.eos_token)[0]
                        cleaned.append(t.strip())
                    decoded = cleaned
                else:
                    decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    
                grouped = [decoded[i:i + self.num_sequences] for i in range(0, len(decoded), self.num_sequences)]

            results.extend(grouped)


        return results