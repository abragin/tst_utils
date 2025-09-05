from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from tst_utils.eval.model_names import GPT_SMALL_TST_MODEL_NAME
from tst_utils.tst_generator import TSTGenerator, GEN_OPTS_V1


class TstViaGPTSmall:
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained(GPT_SMALL_TST_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            GPT_SMALL_TST_MODEL_NAME
        ).to('cuda')
        author_tags = tokenizer.special_tokens_map['additional_special_tokens']
        styles = [
            a_t[1:-1].capitalize()
            for a_t in author_tags
        ]
        self.tst_generator = TSTGenerator(
            model,
            tokenizer,
            target_styles=styles,
            model_type="GPT",
            batch_size=8,
            generate_options=GEN_OPTS_V1,
            max_length=512,
        )

    def __call__(self, texts, target_style):
        return self.tst_generator.perform_tst(texts, target_style)
