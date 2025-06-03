def a2tag(author):
    return f"<{author.lower().replace(' ', '_')}>"

def print_item(inputs, tokenizer):
    source = tokenizer.decode(inputs['input_ids'])
    print(source)
    if 'labels' in inputs:
        target = tokenizer.decode(inputs['labels'])
        print(target)

def print_item_with_debug(item, tokenizer):
    inputs = item[0]
    db_info = item[1]
    print_item(inputs, tokenizer)
    print(db_info)