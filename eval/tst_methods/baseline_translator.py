from transformers import AutoTokenizer, MarianMTModel
from tqdm import tqdm
from tst_utils.eval.model_names import (
    EN_RU_DOSTOEVSKY_MODEL_NAME, EN_RU_TOLSTOY_MODEL_NAME,
    RU_EN_MODEL_NAME
)


def batch_translate(texts, model, tokenizer, cache=None, bs=10):
    """
    Translates a list of texts using a HuggingFace model with optional caching.
    
    Args:
        texts (List[str]): Input texts.
        model: HuggingFace translation model.
        tokenizer: Corresponding tokenizer.
        cache (dict): Optional cache for memoization.
        bs (int): Batch size.
        
    Returns:
        List[str]: Translated texts, preserving original order.
    """
    if cache is None:
        cache = {}  # New dictionary for every call
    new_texts = [text for text in texts if text not in cache]
    if new_texts:
        texts_par = [list(new_texts[i:i + bs]) for i in range(0, len(new_texts), bs)]
        for ts in tqdm(texts_par):
            model_input = tokenizer(
                ts, return_tensors='pt', padding=True, truncation=True, max_length=512
            ).to('cuda')
            out = model.generate(**model_input)
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            for text, trans in zip(ts, decoded):
                cache[text] = trans  # Store result in cache

    return [cache[text] for text in texts]  # Retrieve from cache

class TstViaTranslate:
    def __init__(self):
        self.ru_en_cache = {}
        self.tokenizer_ru_en = AutoTokenizer.from_pretrained(RU_EN_MODEL_NAME)
        self.en_ru_tokenizer = AutoTokenizer.from_pretrained(
            EN_RU_TOLSTOY_MODEL_NAME
        )

    def __call__(self, texts, target_style):
        model_ru_en = MarianMTModel.from_pretrained(RU_EN_MODEL_NAME).to('cuda')
        en_translations = batch_translate(
            texts, model_ru_en, self.tokenizer_ru_en, self.ru_en_cache
        )
        del model_ru_en
        if target_style == 'Tolstoy':
            model_tolstoy =  MarianMTModel.from_pretrained(
                EN_RU_TOLSTOY_MODEL_NAME
            ).to('cuda')
            styled_texts = batch_translate(en_translations, model_tolstoy, self.en_ru_tokenizer)
            del model_tolstoy
        elif target_style == 'Dostoevsky':
            model_dostoevsky =  MarianMTModel.from_pretrained(
                EN_RU_DOSTOEVSKY_MODEL_NAME
            ).to('cuda')
            styled_texts = batch_translate(en_translations, model_dostoevsky, self.en_ru_tokenizer)
            del model_dostoevsky
        else:
            raise Exception("Unsupported style: ", target_style)

        return styled_texts
