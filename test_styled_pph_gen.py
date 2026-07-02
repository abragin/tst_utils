"""Unit tests for the 2A.4 Phase-1 filter logic and target-norm plumbing.

These are GPU-free: they exercise the pure-Python/numpy pieces of
``styled_pph_gen`` that decide *which rows survive* and *what norm the target
style embeddings get* — the two places a silent regression would corrupt the
downstream training data without any error.

Covered:
- ``phase1_keep_mask``: every threshold direction, the activated-only gender
  gate, boundary values, and the NaN / missing-column guards.
- ``_rescale_targets_to_source_norms`` / ``produce_target_style``: unit-norm
  (folder-14), legacy source-norm, and custom target_norm.
- ``save_tst_results``: the required-column assertion.
"""
import numpy as np
import pandas as pd
import pytest

from tst_utils.styled_pph_gen import (
    phase1_keep_mask,
    _rescale_targets_to_source_norms,
    produce_target_style,
    produce_target_style_other_domain,
    produce_target_style_2_other_domains,
    add_target_style_emb,
    compute_domain_target_weights,
    save_tst_results,
    get_target_styles,
    DOMAIN_ONLY,
    DOMAIN_CENTROID_SIM,
    TARGET_WEIGHT_FLOOR,
    TARGET_WEIGHT_TAU,
    CHRF_UB, BI_SCORE_UB, CL_SCORE_UB, GENDER_SCORE_LB,
    ENTITY_SCORE_LB, MEANING_SCORE_LB, SIM_MEASURE_UB, NAT_V2_LB,
)
from tst_utils.eval.metrics.composite import (
    AUTHOR_CE, DOMAIN_CE, _lookup_target_ce,
)


# ---------------- 2A.8.1b: domain -> target-style routing ----------------

# The 9-domain pool_v2 taxonomy. Only 'writers' carries per-author style targets;
# every other domain is domain-only (news pattern). ficbook/taiga_proza/pikabu
# were demoted to domain-only by the 2A.8.2 author-separability diagnostic;
# 'political' carries author names but is domain-targets-only.
REAL_AUTHOR_DOMAINS = ['writers']
DOMAIN_ONLY_DOMAINS = ['news', 'wikipedia', 'taiga_magazines', 'Bible', 'political',
                       'ficbook', 'taiga_proza', 'pikabu']


def test_domain_only_set_matches_taxonomy():
    assert DOMAIN_ONLY == set(DOMAIN_ONLY_DOMAINS)
    # the two groups partition the 9 domains, no overlap
    assert set(REAL_AUTHOR_DOMAINS) & DOMAIN_ONLY == set()
    assert len(REAL_AUTHOR_DOMAINS) + len(DOMAIN_ONLY_DOMAINS) == 9


@pytest.mark.parametrize('domain', REAL_AUTHOR_DOMAINS)
def test_real_author_domains_get_author_and_domain_targets(domain):
    # explicit-flag path (what PphGenerator passes) and the None default path
    # must agree, and must include both author and domain target styles.
    for ts in (
        get_target_styles(domain, has_author_targets=(domain not in DOMAIN_ONLY)),
        get_target_styles(domain),  # None default derives from DOMAIN_ONLY
    ):
        assert any(t.startswith('other_author') for t in ts)
        assert any(t.startswith('other_domain') for t in ts)


@pytest.mark.parametrize('domain', DOMAIN_ONLY_DOMAINS)
def test_domain_only_domains_get_no_author_targets(domain):
    for ts in (
        get_target_styles(domain, has_author_targets=(domain not in DOMAIN_ONLY)),
        get_target_styles(domain),
    ):
        assert not any(t.startswith('other_author') for t in ts)
        assert any(t.startswith('other_domain') for t in ts)


@pytest.mark.parametrize('domain', REAL_AUTHOR_DOMAINS + DOMAIN_ONLY_DOMAINS)
def test_each_domain_name_resolves_to_a_ce_reference(domain):
    # The scoring path passes the *domain name* as target_style for these
    # fallbacks; every one of the 9 domains must resolve to a CE value (R2).
    assert _lookup_target_ce(domain) is not None


def test_new_real_author_domains_fall_back_to_domain_ce():
    # pikabu / taiga_proza usernames are NOT in AUTHOR_CE and never will be; they
    # resolve via DOMAIN_CE through the ficbook-style fallback (composite.py).
    for domain in ('pikabu', 'taiga_proza'):
        assert domain not in AUTHOR_CE
        assert _lookup_target_ce(domain) == DOMAIN_CE[domain]


def _passing_row(**over):
    """A row comfortably inside every keep threshold; override one to fail it."""
    base = dict(
        chrf=0.50, bi_score=0.05, cl_score=0.05,
        gender_score=0.90, gender_activated=True,
        entity_score=0.80, meaning_score=0.85,
        tst_result_style_sim=0.50, nat_v2=0.80,
    )
    base.update(over)
    return base


def _mask(*rows):
    return phase1_keep_mask(pd.DataFrame(list(rows))).to_numpy()


# ---------------- phase1_keep_mask: threshold directions ----------------

def test_clean_row_is_kept():
    assert _mask(_passing_row()).tolist() == [True]


@pytest.mark.parametrize("col,fail_val", [
    ("chrf", CHRF_UB),               # keep is strictly <, so == fails
    ("bi_score", BI_SCORE_UB),
    ("cl_score", CL_SCORE_UB),
    ("tst_result_style_sim", SIM_MEASURE_UB),
    ("entity_score", ENTITY_SCORE_LB - 0.01),
    ("meaning_score", MEANING_SCORE_LB),   # keep is strictly >, so == fails
    ("nat_v2", NAT_V2_LB - 0.01),
])
def test_each_filter_rejects(col, fail_val):
    assert _mask(_passing_row(**{col: fail_val})).tolist() == [False]


@pytest.mark.parametrize("col,keep_val", [
    ("entity_score", ENTITY_SCORE_LB),   # >= so boundary is kept
    ("nat_v2", NAT_V2_LB),               # >= so boundary is kept
])
def test_inclusive_lower_bounds_keep_boundary(col, keep_val):
    assert _mask(_passing_row(**{col: keep_val})).tolist() == [True]


# ---------------- gender gate is activated-only ----------------

def test_gender_rejects_only_when_activated():
    # low gender score but NOT activated -> gate does not fire -> kept
    not_activated = _passing_row(gender_score=0.1, gender_activated=False)
    activated = _passing_row(gender_score=0.1, gender_activated=True)
    assert _mask(not_activated).tolist() == [True]
    assert _mask(activated).tolist() == [False]


def test_gender_activated_above_threshold_kept():
    assert _mask(_passing_row(gender_score=GENDER_SCORE_LB,
                              gender_activated=True)).tolist() == [True]


# ---------------- NaN / missing-column guards ----------------

def test_nan_in_plain_column_raises():
    with pytest.raises(ValueError, match="NaN"):
        _mask(_passing_row(meaning_score=np.nan))


def test_nan_gender_score_on_nonactivated_is_ok():
    # NaN gender_score is legitimate when the pair is not gender-activated
    row = _passing_row(gender_score=np.nan, gender_activated=False)
    assert _mask(row).tolist() == [True]


def test_nan_gender_score_on_activated_raises():
    with pytest.raises(ValueError, match="gender_score"):
        _mask(_passing_row(gender_score=np.nan, gender_activated=True))


def test_missing_column_raises():
    df = pd.DataFrame([_passing_row()]).drop(columns=["nat_v2"])
    with pytest.raises(KeyError, match="nat_v2"):
        phase1_keep_mask(df)


def test_mixed_batch_keeps_only_clean_rows():
    mask = _mask(
        _passing_row(),                       # keep
        _passing_row(bi_score=0.5),           # reject (boilerplate)
        _passing_row(meaning_score=0.1),      # reject (meaning collapse)
        _passing_row(gender_score=0.1, gender_activated=False),  # keep
    )
    assert mask.tolist() == [True, False, False, True]


# ---------------- target_norm plumbing ----------------

def test_rescale_unit_norm():
    src = [np.array([3.0, 4.0]), np.array([0.0, 5.0])]       # norms 5, 5
    tgt = np.array([[1.0, 1.0], [2.0, 0.0]])
    out = _rescale_targets_to_source_norms(src, tgt, target_norm=1.0)
    for v in out:
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_rescale_legacy_matches_source_norm():
    src = [np.array([3.0, 4.0]), np.array([6.0, 8.0])]       # norms 5, 10
    tgt = np.array([[1.0, 1.0], [1.0, 0.0]])
    out = _rescale_targets_to_source_norms(src, tgt, target_norm=None)
    assert np.isclose(np.linalg.norm(out[0]), 5.0)
    assert np.isclose(np.linalg.norm(out[1]), 10.0)


def test_rescale_custom_norm():
    src = [np.array([1.0, 0.0])]
    tgt = np.array([[3.0, 4.0]])
    out = _rescale_targets_to_source_norms(src, tgt, target_norm=15.0)
    assert np.isclose(np.linalg.norm(out[0]), 15.0)


def test_rescale_preserves_direction():
    src = [np.array([10.0, 0.0])]
    tgt = np.array([[3.0, 4.0]])                              # direction (0.6, 0.8)
    out = _rescale_targets_to_source_norms(src, tgt, target_norm=1.0)
    assert np.allclose(out[0], [0.6, 0.8])


def _style_pool():
    rng = np.random.default_rng(0)
    rows = []
    for author in ["a", "b", "c"]:
        for _ in range(3):
            rows.append({"author": author,
                         "text_style_emb": rng.normal(size=8) * 7.0})
    return pd.DataFrame(rows)


def test_produce_target_style_unit_norm_for_folder14():
    pool = _style_pool()
    short = pool[pool.author == "a"].copy()
    out = produce_target_style(short, pool, key_col="author",
                               rng=np.random.default_rng(1), target_norm=1.0)
    for v in out:
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_produce_target_style_legacy_keeps_source_norm():
    pool = _style_pool()
    short = pool[pool.author == "a"].copy()
    src_norms = [np.linalg.norm(e) for e in short.text_style_emb]
    out = produce_target_style(short, pool, key_col="author",
                               rng=np.random.default_rng(1), target_norm=None)
    for v, n in zip(out, src_norms):
        assert np.isclose(np.linalg.norm(v), n)


# ---------------- 2A.8.2: per-source target-domain weighting ----------------

ALL_DOMAINS = list(DOMAIN_CENTROID_SIM.keys())


def test_domain_weights_normalized_and_exclude_self():
    for s in ALL_DOMAINS:
        w = compute_domain_target_weights(s)
        assert s not in w                              # own key excluded
        assert set(w) == set(ALL_DOMAINS) - {s}        # every other domain present
        assert np.isclose(sum(w.values()), 1.0)        # probability vector


def test_domain_weights_floor_keeps_all_positive():
    # Every candidate must be strictly positive so n_picks=2 (replace=False)
    # always has >= 2 non-zero entries. The nearest neighbor is the smallest.
    for s in ALL_DOMAINS:
        w = compute_domain_target_weights(s)
        assert all(v > 0 for v in w.values())


def test_domain_weights_downweight_near_neighbors():
    # Penalty-only: the nearest register is suppressed to the minimum; distant
    # registers sit at the uniform baseline (NOT boosted).
    w = compute_domain_target_weights('news')
    assert min(w, key=w.get) == 'wikipedia'            # news's nearest (0.965)
    # symmetric flagged pair: ficbook's nearest is pikabu.
    wf = compute_domain_target_weights('ficbook')
    assert min(wf, key=wf.get) == 'pikabu'             # ficbook's nearest (0.943)


def test_domain_weights_penalty_only_distant_at_baseline():
    # Distant/outlier domains (below tau) must NOT be boosted above the uniform
    # baseline — Bible (news sim 0.431) sits with the other un-penalized domains,
    # not as a lone maximum (the old reward-distance formula gave it ~0.27).
    w = compute_domain_target_weights('news')
    unpenalized = [t for t in w if DOMAIN_CENTROID_SIM['news'][t] <= TARGET_WEIGHT_TAU]
    assert 'Bible' in unpenalized
    # all un-penalized domains share one baseline weight (renormalized uniform)
    vals = [w[t] for t in unpenalized]
    assert max(vals) - min(vals) < 1e-9
    # and the near-neighbor is strictly below that baseline
    assert w['wikipedia'] < min(vals)


def test_domain_weights_unknown_domain_returns_none():
    assert compute_domain_target_weights('nonexistent') is None


def _domain_pool():
    rng = np.random.default_rng(0)
    rows = []
    for dom in ALL_DOMAINS:
        for _ in range(4):
            rows.append({"domain": dom,
                         "text_style_emb": rng.normal(size=8) * 7.0})
    return pd.DataFrame(rows)


def test_weighted_draw_biases_toward_low_similarity():
    # Empirically, weighted target-domain selection should pick down-weighted
    # neighbors far less than favored distant registers.
    pool = _domain_pool()
    short = pd.concat([pool[pool.domain == 'news'].iloc[[0]]] * 2000, ignore_index=True)
    w = compute_domain_target_weights('news')
    rng = np.random.default_rng(7)
    # Reproduce the which-key draw the function performs internally.
    candidates = [d for d in ALL_DOMAINS if d != 'news']
    p = np.array([w[c] for c in candidates]); p = p / p.sum()
    picks = rng.choice(candidates, size=20000, p=p)
    counts = pd.Series(picks).value_counts(normalize=True)
    assert counts['wikipedia'] < counts['Bible']       # near neighbor suppressed
    assert counts['wikipedia'] < 0.05                  # ~0.9% expected


def test_weighted_n_picks_2_no_crash():
    # The positive floor guarantees >= 2 non-zero entries, so the n_picks=2
    # replace=False draw never raises "fewer non-zero entries in p than size".
    pool = _domain_pool()
    short = pool[pool.domain == 'news'].copy()
    out = produce_target_style_2_other_domains(
        short, pool, rng=np.random.default_rng(1), target_norm=1.0,
        domain_weights=compute_domain_target_weights('news'),
    )
    assert len(out) == len(short)
    for v in out:
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_weighted_draw_reproducible_and_differs_from_uniform():
    # Same seed -> identical output (new weighted stream is deterministic);
    # weighting shifts the stream vs uniform, so results differ.
    pool = _domain_pool()
    short = pool[pool.domain == 'news'].copy()
    w = compute_domain_target_weights('news')
    a = produce_target_style_other_domain(
        short, pool, rng=np.random.default_rng(3), target_norm=1.0, domain_weights=w)
    b = produce_target_style_other_domain(
        short, pool, rng=np.random.default_rng(3), target_norm=1.0, domain_weights=w)
    u = produce_target_style_other_domain(
        short, pool, rng=np.random.default_rng(3), target_norm=1.0, domain_weights=None)
    assert all(np.allclose(x, y) for x, y in zip(a, b))          # reproducible
    assert not all(np.allclose(x, y) for x, y in zip(a, u))      # differs from uniform


def test_2domain_mixes_stay_uniform_wiring():
    # 2A.8.2 decision: per-source weighting applies ONLY to single-domain
    # (other_domain) selection; 2-domain weighted-average mixes stay uniform
    # because a mixture's closeness to source isn't monotonic in its components'.
    # This guards add_target_style_emb's routing so a future refactor can't
    # silently forward domain_weights to the 2-domain path.
    from unittest import mock
    import tst_utils.styled_pph_gen as m

    df = pd.DataFrame({
        'text_style_emb': [np.ones(4) * 7.0, np.ones(4) * 7.0],
        'domain': ['news', 'news'],
        'target_style_desc': ['other_domain', '2_domains_weighted_avg'],
    })
    weights = compute_domain_target_weights('news')

    def fake(df_sub, *a, **k):
        embs = [np.zeros(4) for _ in range(len(df_sub))]
        # add_target_style_emb calls producers with return_keys=True and unpacks
        # (embs, keys); mirror that contract so the routing assertions still run.
        if k.get('return_keys'):
            return embs, [['x'] for _ in range(len(df_sub))]
        return embs

    with mock.patch.object(m, 'produce_target_style_other_domain', side_effect=fake) as p1, \
         mock.patch.object(m, 'produce_target_style_2_other_domains', side_effect=fake) as p2:
        m.add_target_style_emb(df, _domain_pool(), domain_weights=weights)

    # single-domain path receives the weights; 2-domain path must NOT
    assert p1.call_args.kwargs.get('domain_weights') == weights
    assert p2.call_args.kwargs.get('domain_weights') is None


def test_degenerate_all_zero_weights_falls_back_to_uniform():
    # A weight map that zeroes every candidate for a row must not crash; the
    # function drops to uniform sampling for that row.
    pool = _domain_pool()
    short = pool[pool.domain == 'news'].copy()
    out = produce_target_style_other_domain(
        short, pool, rng=np.random.default_rng(1), target_norm=1.0,
        domain_weights={d: 0.0 for d in ALL_DOMAINS},
    )
    assert len(out) == len(short)


# ---------------- save_tst_results required-column guard ----------------

def test_save_tst_results_missing_required_raises(tmp_path):
    df = pd.DataFrame([{"styled_text": "x"}])   # missing most required cols
    with pytest.raises(KeyError, match="required columns missing"):
        save_tst_results(df, str(tmp_path) + "/")


def _minimal_valid_tst_df(with_keys=True):
    # A frame carrying every REQUIRED_PERSIST_COLS value so save_tst_results does
    # not raise; two rows with n_picks 1 and 2 target_keys.
    row = {
        'text': 'src', 'target_style_desc': 'other_domain',
        'target_style_sim_measure': 0.5, 'styled_text': 'out',
        'styled_text_style_emb': np.ones(4, dtype=np.float32),
        'meaning_score': 0.9, 'tst_result_style_sim': 0.3,
        'bi_score': 0.1, 'cl_score': 0.1, 'gender_score': 0.9,
        'gender_activated': False, 'entity_score': 1.0, 'chrf': 0.2,
        'nat_v2': 0.5, 'naturality_score': 0.0, 'style_score': 0.0, 'score': 0.0,
    }
    r1 = dict(row); r1['target_keys'] = ['Bible']
    r2 = dict(row, target_style_desc='2_domains_weighted_avg')
    r2['target_keys'] = ['Bible', 'news']
    df = pd.DataFrame([r1, r2])
    if not with_keys:
        df = df.drop(columns=['target_keys'])
    return df


def test_save_tst_results_persists_target_keys(tmp_path):
    df = _minimal_valid_tst_df(with_keys=True)
    save_tst_results(df, str(tmp_path) + "/")
    back = pd.read_parquet(str(tmp_path) + "/part_00001.parquet.gzip")
    assert 'target_keys' in back.columns
    assert back['target_keys'].notna().all()
    # parquet returns an ndarray, not a list — coerce before comparing (review #5)
    assert [list(v) for v in back['target_keys']] == [['Bible'], ['Bible', 'news']]


def test_save_tst_results_missing_target_keys_raises(tmp_path):
    # target_keys is now a REQUIRED_PERSIST_COL: an absent column means the
    # upstream whitelist dropped provenance — fail loud, don't persist blind.
    df = _minimal_valid_tst_df(with_keys=False)
    with pytest.raises(KeyError, match="required columns missing"):
        save_tst_results(df, str(tmp_path) + "/")


# ---------------- target_keys provenance ----------------

def test_return_keys_off_preserves_list_contract():
    # Default return_keys=False must keep the plain list-return that every
    # existing caller (notebooks, add_styled_pph*.py) relies on.
    pool = _style_pool()
    short = pool[pool.author == "a"].copy()
    out = produce_target_style(short, pool, key_col="author",
                               rng=np.random.default_rng(1), target_norm=1.0)
    assert isinstance(out, list)
    assert all(isinstance(v, np.ndarray) for v in out)


def test_return_keys_shape_and_membership_n_picks_1():
    pool = _style_pool()
    short = pool[pool.author == "a"].copy()
    embs, keys = produce_target_style(
        short, pool, key_col="author", n_picks=1,
        rng=np.random.default_rng(1), target_norm=1.0, return_keys=True)
    assert len(keys) == len(embs) == len(short)
    candidates = set(pool.author) - {"a"}
    for row_keys in keys:
        assert len(row_keys) == 1                     # length == n_picks
        assert "a" not in row_keys                    # own key excluded
        assert set(row_keys) <= candidates            # drawn from candidate set


def test_return_keys_shape_and_membership_n_picks_2():
    pool = _style_pool()
    short = pool[pool.author == "a"].copy()
    embs, keys = produce_target_style(
        short, pool, key_col="author", n_picks=2,
        rng=np.random.default_rng(2), target_norm=1.0, return_keys=True)
    candidates = set(pool.author) - {"a"}
    for row_keys in keys:
        assert len(row_keys) == 2                     # length == n_picks
        assert "a" not in row_keys                    # own key excluded
        assert len(set(row_keys)) == 2                # distinct (replace=False)
        assert set(row_keys) <= candidates


def test_return_keys_reproducible_unweighted():
    # Same seed -> identical keys on the unweighted path.
    pool = _domain_pool()
    short = pool[pool.domain == 'news'].copy()
    _, k1 = produce_target_style_other_domain(
        short, pool, rng=np.random.default_rng(5), target_norm=1.0, return_keys=True)
    _, k2 = produce_target_style_other_domain(
        short, pool, rng=np.random.default_rng(5), target_norm=1.0, return_keys=True)
    assert [list(r) for r in k1] == [list(r) for r in k2]


def test_return_keys_reproducible_weighted_and_differs_from_uniform():
    # The weighted draw is deterministic under a fixed seed, but diverges the RNG
    # bit stream from the unweighted path (2A.8.2 note) — so the *keys* it yields
    # differ from the uniform draw at the same seed. This re-baselines the
    # per-path expectation without pinning the fragile internal candidate order.
    pool = _domain_pool()
    short = pd.concat([pool[pool.domain == 'news']] * 5, ignore_index=True)
    w = compute_domain_target_weights('news')
    _, a = produce_target_style_other_domain(
        short, pool, rng=np.random.default_rng(9), target_norm=1.0,
        domain_weights=w, return_keys=True)
    _, b = produce_target_style_other_domain(
        short, pool, rng=np.random.default_rng(9), target_norm=1.0,
        domain_weights=w, return_keys=True)
    _, u = produce_target_style_other_domain(
        short, pool, rng=np.random.default_rng(9), target_norm=1.0,
        domain_weights=None, return_keys=True)
    ak = [list(r) for r in a]
    assert ak == [list(r) for r in b]                 # reproducible
    assert ak != [list(r) for r in u]                 # stream differs from uniform


def _prov_pool():
    # A pool with both author and domain columns so add_target_style_emb can
    # route every target_style_desc branch.
    rng = np.random.default_rng(0)
    rows = []
    for dom, authors in [('lit', ['Tolstoy', 'Chekhov', 'Dostoevsky'])]:
        for a in authors:
            for _ in range(4):
                rows.append({"author": a, "domain": dom,
                             "text_style_emb": rng.normal(size=8) * 7.0})
    for dom in ['news', 'Bible', 'wikipedia']:
        for _ in range(4):
            rows.append({"author": dom, "domain": dom,
                         "text_style_emb": rng.normal(size=8) * 7.0})
    return pd.DataFrame(rows)


def test_add_target_style_emb_writes_keys_every_branch():
    pool = _prov_pool()
    in_domain = pool[pool.domain == 'lit']
    descs = ['other_author', '2_authors_weighted_avg',
             'other_domain', '2_domains_weighted_avg',
             'other_author_with_noise', 'other_domain_with_negations']
    # one source row per desc, all from author Tolstoy / domain lit
    df = in_domain[in_domain.author == 'Tolstoy'].iloc[[0]].copy()
    df = pd.concat([df] * len(descs), ignore_index=True)
    df['target_style_desc'] = descs

    add_target_style_emb(df, pool, in_domain_style_df=in_domain,
                         target_norm=1.0, rng=np.random.default_rng(3))

    assert 'target_keys' in df.columns
    assert df['target_keys'].notna().all()            # written for every branch
    for _, row in df.iterrows():
        desc = row.target_style_desc
        keys = list(row.target_keys)
        # cross-alignment: key count matches the mechanism implied by the desc
        n_expected = 2 if desc.startswith('2_') else 1
        assert len(keys) == n_expected
        # own author/domain never appears in its own keys
        assert 'Tolstoy' not in keys                  # own author excluded
        assert 'lit' not in keys                      # own domain excluded
        # author vs domain namespace matches the mechanism
        if 'author' in desc:
            assert all(k in {'Chekhov', 'Dostoevsky'} for k in keys)
        else:
            assert all(k in {'news', 'Bible', 'wikipedia'} for k in keys)
