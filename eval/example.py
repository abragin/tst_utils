from tst_utils.eval.data.load import load_test_df, load_llm_data, load_author_styles
from tst_utils.eval.tst_methods.baseline_translator import TstViaTranslate
from tst_utils.eval.tst_methods.gpt_small_tst import tst_via_gptsmall
from tst_utils.eval.performance import TstPerformanceMetrics, TARGET_STYLES


def run_translate_eval(test_chunks, author_styles):
    print("\n--- Running Translate-based TST Evaluation ---")
    pm = TstPerformanceMetrics(
        test_df=test_chunks,
        tst_func=TstViaTranslate(),
        target_styles=TARGET_STYLES['DT'],
        tst_model='translate',
        author_styles=author_styles,
        verbose=True
    )
    pm.produce_tst_results()
    pm.compute_scores()
    print("\nPerformance (Translate):")
    print(pm.final_performance())
    return pm


def run_gpt_small_eval(test_chunks, author_styles):
    print("\n--- Running GPT-Small-based TST Evaluation ---")
    pm = TstPerformanceMetrics(
        test_df=test_chunks,
        tst_func=tst_via_gptsmall,
        target_styles=TARGET_STYLES['BCDT'],
        tst_model='gpt_small',
        author_styles=author_styles,
        verbose=True
    )
    pm.produce_tst_results()
    pm.compute_scores()
    print("\nPerformance (GPT Small):")
    print(pm.final_performance())
    return pm


def run_llm_eval(test_chunks, author_styles, llm_results):
    print("\n--- Evaluating Pre-generated LLM Results ---")
    pm = TstPerformanceMetrics(
        test_df=test_chunks,
        tst_func=lambda _: None,
        target_styles=TARGET_STYLES['COMPLETE'],
        tst_model='llms',
        author_styles=author_styles,
        verbose=True
    )
    pm.tst_results = llm_results
    pm.compute_scores()
    print("\nPerformance (All LLMs):")
    print(pm.final_performance())
    return pm


def run_specific_llm_eval(test_chunks, author_styles, all_llm_results, model_name='chatgpt'):
    print(f"\n--- Evaluating Specific LLM Results: {model_name} ---")
    pm = TstPerformanceMetrics(
        test_df=test_chunks,
        tst_func=lambda _: None,
        target_styles=TARGET_STYLES['COMPLETE'],
        tst_model=model_name,
        author_styles=author_styles,
        verbose=True
    )
    pm.tst_results = all_llm_results[all_llm_results.llm == model_name].copy()
    print(f"\nPerformance ({model_name}):")
    print(pm.final_performance())
    return pm


def main():
    test_chunks = load_test_df()
    llm_results = load_llm_data()
    author_styles = load_author_styles()

    _ = run_translate_eval(test_chunks, author_styles)
    _ = run_gpt_small_eval(test_chunks, author_styles)
    all_llms_pm = run_llm_eval(test_chunks, author_styles, llm_results)
    _ = run_specific_llm_eval(test_chunks, author_styles, all_llms_pm.tst_results, model_name='chatgpt')


if __name__ == "__main__":
    main()