from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from tst_utils.eval.model_names import GPT_SMALL_TST_MODEL_NAME


def generate_with_style(text, author_tag, model, tokenizer):
    """
    Generate styled text using a conditional GPT model.

    Args:
        text (str): Input text.
        author_tag (str): Style tag, e.g., '<tolstoy>'.
        model: Pretrained CausalLM.
        tokenizer: Corresponding tokenizer.

    Returns:
        str: Generated styled continuation.
    """
    input_ids = tokenizer(text + f" {author_tag}", return_tensors="pt").to('cuda').input_ids
    num_sequences = 1 #@param {type:"integer"}
    min_length =  50 #@param {type:"integer"}
    max_length =   1024#@param {type:"integer"}
    temperature = 0.8 #@param {type:"slider", min:0, max:3, step:0.01}
    top_p = 0.95 #@param {type:"slider", min:0, max:1, step:0.01}
    top_k = 50 #@param {type:"integer"}
    repetition_penalty =  1.0#@param {t
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        min_length=min_length,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        do_sample=True,
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_sequences
    )
    return [
        t.split(author_tag)[1].split('</s>')[0][1:]
        for t in tokenizer.batch_decode(output_sequences)
    ][0]

def tst_via_gptsmall(texts, target_style):
    tokenizer = AutoTokenizer.from_pretrained(GPT_SMALL_TST_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(GPT_SMALL_TST_MODEL_NAME
            ).to('cuda')
    author_tags = tokenizer.special_tokens_map['additional_special_tokens']
    styles = [
        a_t[1:-1].capitalize()
        for a_t in author_tags
    ]
    if target_style not in styles:
        raise Exception("Unsupported style: ", target_style)
    author_tag = f"<{target_style.lower()}>"
    return [generate_with_style(t, author_tag, model, tokenizer) for t in tqdm(texts)]
