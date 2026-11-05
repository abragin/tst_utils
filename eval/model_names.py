RU_EN_MODEL_NAME = 'Helsinki-NLP/opus-mt-ru-en'
EN_RU_DOSTOEVSKY_MODEL_NAME = "abragin/opus-mt-en-ru-ft-dostoevsky"
EN_RU_TOLSTOY_MODEL_NAME = "abragin/opus-mt-en-ru-ft-tolstoy"
GPT_SMALL_TST_MODEL_NAME = "abragin/rugpt3small_tst"
LABSE_MODEL_NAME = "cointegrated/LaBSE-en-ru"
# The Hub commit the LaBSE loads pin. Measured on tallin on 2026-10-06: the only
# snapshot in the cache, and the value the unpinned load already resolved.
LABSE_REVISION = "cf0714e606d4af551e14ad69a7929cd6b0da7f7e"
PERPL_MODEL_NAME = 'ai-forever/rugpt3small_based_on_gpt2'
STYLE_MODEL = "abragin/ruBert-style-v2"
# The entry of style_encoders.json that STYLE_MODEL names.
STYLE_ENCODER_KEY = "base_v2"


def style_encoder_key():
    """The pinned style encoder key, read at call time (task 5.2).

    This is the one pin source. Readers call it whenever they need the
    default encoder instead of copying `STYLE_ENCODER_KEY` at import or
    definition time, so a patch of `model_names.STYLE_ENCODER_KEY` (a test,
    or the pin commit) reaches every reader.
    """
    return STYLE_ENCODER_KEY
