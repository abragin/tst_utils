from tst_utils.datasets import ChapterDataset
import pandas as pd
from transformers import AutoTokenizer


min_tok_len = 4
avg_tok_len = 7
max_tok_len = 11
max_length= 512
author_tags = ['<news>']
model_name_t5 = "ai-forever/ruT5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_t5)
tokenizer.add_special_tokens({'additional_special_tokens': author_tags})
model_name_gpt = "ai-forever/rugpt3small_based_on_gpt2"
tokenizer_gpt =  AutoTokenizer.from_pretrained(model_name_gpt)
tokenizer_gpt.add_special_tokens({'additional_special_tokens': author_tags})


test_df = pd.DataFrame({
'text_ru': [
    'Первый текст - оригинал.',
    'Второй текст - оригинал.',
    'Третий текст - оригинал.',
],
'text_source_opt1': [
    'Первый текст - первая версия.',
    'Второй текст - первая версия.',
    'Третий текст - первая версия.'
],
'text_source_opt2': [
    'Первый текст - вторая версия.',
    'Второй текст - вторая версия.',
    'Третий текст - вторая версия.'
]
})
test_df['author'] = 'News'

def test_with_1_source_col():
    dataset = ChapterDataset(
        test_df,
        tokenizer=tokenizer,
        source_cols = ['text_source_opt1'],
        model_type='T5',
        min_tok_len=min_tok_len,
        avg_tok_len=avg_tok_len,
        max_tok_len=max_tok_len,
        max_length=max_length
    )
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"
    assert dataset.source_options == [['text_source_opt1']] * 3, "Incorrect `source_options` value"
    assert (dataset.selected_options) == ['text_source_opt1'] * 3, "Unexpected value of select_options"
    dataset.segment_ranges = [(0, 1), (1, 3)]
    source_col = test_df.text_source_opt1
    assert (
        tokenizer.decode(dataset[0]['input_ids']) == "<news> " + ' '.join(source_col[0:1]) + "</s>"
    ), "Incorrect source input_ids values"
    assert (tokenizer.decode(dataset[1]['labels']) == ' '.join(test_df.text_ru[1:3]) + "</s>")

def test_with_2_source_cols_but_1_empty():
    df = test_df.copy()
    df['text_source_opt1'] = ''
    dataset = ChapterDataset(
        df,
        tokenizer=tokenizer,
        source_cols = ['text_source_opt1', 'text_source_opt2'],
        model_type='T5',
        min_tok_len=min_tok_len,
        avg_tok_len=avg_tok_len,
        max_tok_len=max_tok_len,
        max_length=max_length
    )

    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"
    assert dataset.source_options == [['text_source_opt2']] * 3, "Incorrect `source_options` value"
    assert (dataset.selected_options) == ['text_source_opt2'] * 3, "Unexpected value of select_options"
    dataset.segment_ranges = [(0, 2), (2, 3)]
    source_col = test_df.text_source_opt2
    assert (
        tokenizer.decode(dataset[0]['input_ids']) == "<news> " +  ' '.join(source_col[0:2]) + "</s>"
    ), "Incorrect source input_ids values"
    assert (tokenizer.decode(dataset[1]['labels']) == ' '.join(test_df.text_ru[2:3]) + "</s>")

def test_with_2_source_cols():
    dataset = ChapterDataset(
        test_df,
        tokenizer=tokenizer,
        source_cols = ['text_source_opt1', 'text_source_opt2'],
        model_type='T5',
        min_tok_len=min_tok_len,
        avg_tok_len=avg_tok_len,
        max_tok_len=max_tok_len,
        max_length=max_length
    )
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"
    assert (
        dataset.source_options == [['text_source_opt1', 'text_source_opt2']] * 3
    ), "Incorrect `source_options` value"
    dataset.selected_options = ['text_source_opt1', 'text_source_opt2', 'text_source_opt1']
    dataset.segment_ranges = [(0, 1), (1, 3)]
    expected_source = f"<news> {test_df.text_source_opt2[1]} {test_df.text_source_opt1[2]}</s>"
    assert (
        tokenizer.decode(dataset[1]['input_ids']) == expected_source
    ), "Incorrect source input_ids values"
    assert (tokenizer.decode(dataset[0]['labels']) == ' '.join(test_df.text_ru[0:1]) + "</s>")

def test_with_2_source_cols_but_some_recs_are_empty():
    df = test_df.copy()
    df.loc[0,'text_source_opt1'] = ''
    df.loc[1,'text_source_opt2'] = ''

    dataset = ChapterDataset(
        df,
        tokenizer=tokenizer,
        model_type='T5',
        source_cols = ['text_source_opt1', 'text_source_opt2'],
        min_tok_len=min_tok_len,
        avg_tok_len=avg_tok_len,
        max_tok_len=max_tok_len,
        max_length=max_length
    )

    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"
    assert dataset.source_options == [
        ['text_source_opt2'],
        ['text_source_opt1'],
        ['text_source_opt1', 'text_source_opt2']
    ], "Incorrect `source_options` value"
    dataset.selected_options = ['text_source_opt2', 'text_source_opt1', 'text_source_opt1']    
    dataset.segment_ranges = [(0, 1), (1, 3)]
    expected_source = f"<news> {test_df.text_source_opt1[1]} {test_df.text_source_opt1[2]}</s>"
    assert (
        tokenizer.decode(dataset[1]['input_ids']) == expected_source
    ), "Incorrect source input_ids values"
    assert (tokenizer.decode(dataset[0]['labels']) == ' '.join(test_df.text_ru[0:1]) + "</s>")

def test_with_2_source_cols_some_recs_are_empty_and_gpt_model_type():
    df = test_df.copy()
    df.loc[0,'text_source_opt1'] = ''
    df.loc[1,'text_source_opt2'] = ''

    dataset = ChapterDataset(
        df,
        tokenizer=tokenizer_gpt,
        model_type='GPT',
        source_cols = ['text_source_opt1', 'text_source_opt2'],
        min_tok_len=min_tok_len,
        avg_tok_len=avg_tok_len,
        max_tok_len=max_tok_len,
        max_length=max_length
    )

    dataset.selected_options = ['text_source_opt2', 'text_source_opt1', 'text_source_opt1']    
    dataset.segment_ranges = [(0, 1), (1, 3)]
    expected_input = (
        f"{test_df.text_source_opt1[1]} {test_df.text_source_opt1[2]} <news> " +
        ' '.join(test_df.text_ru[1:3]) + tokenizer_gpt.eos_token
    )
    assert (
        tokenizer_gpt.decode(dataset[1]['input_ids']) == expected_input
    ), "Incorrect source input_ids values for GPT model type"


def test_max_length():
    dataset = ChapterDataset(
        test_df,
        tokenizer=tokenizer,
        source_cols = ['text_source_opt1', 'text_source_opt2'],
        model_type = 'T5',
        min_tok_len=min_tok_len,
        avg_tok_len=avg_tok_len,
        max_tok_len=max_tok_len,
        max_length=11
    )
    dataset.segment_ranges = [(0, 2), (2, 3)]
    item = dataset[0]
    assert len(item['input_ids']) == 11, 'expected length of input_ids to be equal to 11'
    assert len(item['labels']) == 11, 'expected length of labels to be equal to 11'
    assert len(item['attention_mask']) == 11, 'expected length of attention_mask to be equal to 11'
