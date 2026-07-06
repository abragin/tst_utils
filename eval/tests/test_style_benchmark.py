"""Tests for tst_utils.eval.style_benchmark on tiny synthetic frames.

No GPU / no model downloads: encoders are injected as dict-lookup embed
functions, assets are synthesized into tmp_path with the frozen schemas.
"""

import numpy as np
import pandas as pd
import pytest

from tst_utils.eval.style_benchmark import (
    BenchmarkAssets, SCORECARD_COLUMNS, bootstrap_auc_ci, fp_adjusted_auc,
    rank_auc, read_scorecard, resolve_embed_fn, rowwise_sim, score_encoder,
    silhouette_by_group, write_scorecard,
)

DIM = 8
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# metric math
# ---------------------------------------------------------------------------

def test_rowwise_sim_matches_scalar_sim_measure():
    from tst_utils.eval.metrics.style import sim_measure
    u = RNG.normal(size=(20, DIM))
    v = RNG.normal(size=(20, DIM))
    got = rowwise_sim(u, v)
    want = [sim_measure(a, b) for a, b in zip(u, v)]
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_rank_auc_known_values():
    assert rank_auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert rank_auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert rank_auc([1, 1, 1], [1, 1, 1]) == 0.5  # all ties -> chance
    # one overlap point shared between classes: (1>0) + (1~1)*0.5 + 2 wins
    assert rank_auc([1, 2], [0, 1]) == pytest.approx(0.875)
    assert np.isnan(rank_auc([], [1.0]))


def test_fp_adjusted_auc():
    assert fp_adjusted_auc(0.8, 0.0) == pytest.approx(0.8)
    assert fp_adjusted_auc(0.5, 0.3) == pytest.approx(0.5)
    # auc_obs = (1-fp)*auc_true + fp*0.5: recover auc_true = 0.9 at fp = 0.2
    assert fp_adjusted_auc(0.9 * 0.8 + 0.5 * 0.2, 0.2) == pytest.approx(0.9)
    assert np.isnan(fp_adjusted_auc(0.8, 1.0))


def test_bootstrap_auc_ci_deterministic_and_ordered():
    pos = RNG.normal(1.0, 1.0, size=40)
    neg = RNG.normal(0.0, 1.0, size=40)
    lo1, hi1 = bootstrap_auc_ci(pos, neg, seed=1, n_boot=200)
    lo2, hi2 = bootstrap_auc_ci(pos, neg, seed=1, n_boot=200)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= rank_auc(pos, neg) <= hi1


# ---------------------------------------------------------------------------
# synthetic assets + injected encoder
# ---------------------------------------------------------------------------

class FakeEncoder:
    """Deterministic text->vector lookup; the injectable encoder contract."""

    def __init__(self):
        self.vectors = {}

    def register(self, text, vec):
        self.vectors[text] = np.asarray(vec, dtype=np.float64)
        return text

    def __call__(self, texts):
        return np.stack([self.vectors[t] for t in texts])


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return vec / np.linalg.norm(vec)


def _make_pair(enc, name, moved):
    """Register a (text, styled_text) pair; ``moved`` -> styled far away."""
    base = _unit(RNG.normal(size=DIM))
    if moved:
        other = _unit(RNG.normal(size=DIM))
        styled = _unit(base + 3.0 * other)
    else:
        styled = _unit(base + 0.05 * RNG.normal(size=DIM))
    return (enc.register(f'{name} src', base),
            enc.register(f'{name} sty', styled))


@pytest.fixture
def synthetic(tmp_path):
    enc = FakeEncoder()

    # --- a_formal: ub_benchmark dir --------------------------------------
    ub_dir = tmp_path / 'ub_benchmark'
    ub_dir.mkdir()
    r1_rows, in_rows = [], []
    pid = 0
    for source in ('news', 'wikipedia'):
        for bucket, n, labels in (
                ('reject', 6, ['transfer'] * 4 + ['paraphrase'] * 2),
                ('rt_paraphrase', 4,
                 ['transfer'] + ['paraphrase'] * 3),  # anchor_fp = 0.25
                ('accepted', 2, ['transfer'] * 2)):
            for i in range(n):
                pid += 1
                label = labels[i]
                moved = (bucket == 'reject' and label == 'transfer')
                text, styled = _make_pair(enc, f'{source}-{bucket}-{i}',
                                          moved)
                r1_rows.append({'pair_id': pid, 'source': source,
                                'bucket': bucket, 'majority4': label})
                in_rows.append({'pair_id': pid, 'text': text,
                                'styled_text': styled})
    pd.DataFrame(r1_rows).to_csv(ub_dir / 'ub_label_analysis.csv',
                                 index=False)
    pd.DataFrame(in_rows).to_csv(ub_dir / 'ub_label_input.csv', index=False)

    val_rows, vin_rows, judge_rows = [], [], []
    for bucket, n, labels in (
            ('window_reject', 5, ['transfer'] * 3 + ['paraphrase'] * 2),
            ('rt_paraphrase', 3, ['paraphrase'] * 3)):  # anchor_fp = 0.0
        for i in range(n):
            pid += 1
            label = labels[i]
            moved = (bucket == 'window_reject' and label == 'transfer')
            text, styled = _make_pair(enc, f'val-{bucket}-{i}', moved)
            val_rows.append({'pair_id': pid, 'bucket': bucket,
                             'chrf': 0.5, 'style_sim': 0.93})
            vin_rows.append({'pair_id': pid, 'text': text,
                             'styled_text': styled})
            judge_rows.append({'pair_id': pid, 'label': label,
                               'comment': ''})
    pd.DataFrame(val_rows).to_csv(ub_dir / 'ub_validation_bucket_map.csv',
                                  index=False)
    pd.DataFrame(vin_rows).to_csv(ub_dir / 'ub_validation_input.csv',
                                  index=False)
    for judge in ('sonnet_val', 'deepseek_val', 'mimo_val'):
        pd.DataFrame(judge_rows).to_csv(
            ub_dir / f'llm_review_{judge}.csv', index=False)

    # --- b1 / b2: paired frames with a real attribute direction ----------
    attr_axis = _unit(RNG.normal(size=DIM))

    def paired_frame(name, n):
        rows = []
        for i in range(n):
            base = _unit(RNG.normal(size=DIM))
            styled = _unit(base + 1.5 * attr_axis
                           + 0.02 * RNG.normal(size=DIM))
            rows.append({
                'text': enc.register(f'{name}-{i} src', base),
                'styled_text': enc.register(f'{name}-{i} sty', styled)})
        return pd.DataFrame(rows)

    pdx = paired_frame('pdx', 40)
    pdx.to_parquet(tmp_path / 'paradetox.parquet')

    gold_parts = []
    for strategy, n in (('b_axis_informal', 40),
                        ('d_near_paradetox_neutral', 20)):
        part = paired_frame(strategy, n).rename(
            columns={'styled_text': 'formal_text'})
        part['strategy'] = strategy
        part['gold_pass'] = True
        gold_parts.append(part)
    pd.concat(gold_parts).to_parquet(tmp_path / 'formality_gold.parquet')

    # --- axis (c): author pairs + chunks + writers reference --------------
    ap_dir = tmp_path / 'author_pairs_benchmark'
    ap_dir.mkdir()
    author_vecs = {f'a{i}': _unit(RNG.normal(size=DIM)) for i in range(6)}

    def author_text(name, aid, j):
        vec = _unit(author_vecs[aid] + 0.1 * RNG.normal(size=DIM))
        return enc.register(f'{name}-{aid}-{j}', vec)

    c_pairs = []
    for i in range(6):
        a1, a2 = f'a{i % 6}', f'a{(i + 1) % 6}'
        c_pairs.append({'kind': 'same_post_diff_author',
                        'post_id': f'p{i}', 'author_id_a': a1,
                        'author_id_b': a2,
                        'text_a': author_text(f'cd{i}', a1, 0),
                        'text_b': author_text(f'cd{i}', a2, 1)})
        c_pairs.append({'kind': 'same_post_same_author',
                        'post_id': f'p{i}', 'author_id_a': a1,
                        'author_id_b': a1,
                        'text_a': author_text(f'cs{i}', a1, 0),
                        'text_b': author_text(f'cs{i}', a1, 1)})
        c_pairs.append({'kind': 'cross_post_same_author',
                        'post_id': f'p{i}', 'author_id_a': a2,
                        'author_id_b': a2,
                        'text_a': author_text(f'cx{i}', a2, 0),
                        'text_b': author_text(f'cx{i}', a2, 1)})
    pd.DataFrame(c_pairs).to_parquet(
        ap_dir / 'pikabu_author_pairs_v1.parquet')

    c_chunks = [{'post_id': f'p{j}', 'author_id': aid,
                 'text': author_text(f'ch{j}', aid, j)}
                for aid in author_vecs for j in range(4)]
    pd.DataFrame(c_chunks).to_parquet(
        ap_dir / 'pikabu_author_chunks_eval.parquet')

    w_pairs = []
    for i in range(6):
        a1, a2 = f'a{i % 6}', f'a{(i + 2) % 6}'
        w_pairs.append({'kind': 'writers_same_author', 'author_a': a1,
                        'author_b': a1,
                        'text_a': author_text(f'ws{i}', a1, 0),
                        'text_b': author_text(f'ws{i}', a1, 1)})
        w_pairs.append({'kind': 'writers_diff_author', 'author_a': a1,
                        'author_b': a2,
                        'text_a': author_text(f'wd{i}', a1, 0),
                        'text_b': author_text(f'wd{i}', a2, 1)})
    pd.DataFrame(w_pairs).to_parquet(
        ap_dir / 'writers_reference_pairs_v1.parquet')

    # --- a_informal: analysis csv + blind input ---------------------------
    ubi_dir = tmp_path / 'ub_benchmark_informal'
    ubi_dir.mkdir()
    ubi_ana, ubi_inp = [], []
    ipid = 6000
    for source in ('ficbook', 'pikabu', 'taiga_proza'):
        for bucket, n, labels in (
                ('reject', 5, ['transfer'] * 3 + ['paraphrase'] * 2),
                ('llm_paraphrase', 4,
                 ['transfer'] + ['paraphrase'] * 3)):  # anchor_fp = 0.25
            for i in range(n):
                ipid += 1
                label = labels[i]
                moved = (bucket == 'reject' and label == 'transfer')
                text, styled = _make_pair(enc, f'inf-{source}-{bucket}-{i}',
                                          moved)
                ubi_ana.append({'pair_id': ipid, 'source': source,
                                'bucket': bucket, 'majority_usable': label})
                ubi_inp.append({'pair_id': ipid, 'text': text,
                                'styled_text': styled})
    pd.DataFrame(ubi_ana).to_csv(ubi_dir / 'informal_label_analysis.csv',
                                 index=False)
    pd.DataFrame(ubi_inp).to_csv(ubi_dir / 'informal_label_input.csv',
                                 index=False)

    assets = BenchmarkAssets(
        ub_formal_dir=str(ub_dir),
        paradetox_path=str(tmp_path / 'paradetox.parquet'),
        formality_gold_path=str(tmp_path / 'formality_gold.parquet'),
        ub_informal_dir=str(ubi_dir),
        author_pairs_dir=str(ap_dir))
    return enc, assets


def test_score_encoder_end_to_end(synthetic):
    enc, assets = synthetic
    card = score_encoder(enc, encoder_id='fake-v0', assets=assets,
                         n_boot=100)

    assert list(card.columns) == SCORECARD_COLUMNS
    assert (card['encoder_id'] == 'fake-v0').all()
    assert card['asset_version'].notna().all()

    # a_formal: 3 slices x 5 metric rows
    a = card[card['axis'] == 'a_formal']
    assert sorted(a['slice'].unique()) == ['news', 'news_validation',
                                           'wikipedia']
    assert len(a) == 15
    # synthetic geometry is fully separable -> AUC 1.0 on every slice
    aucs = a[a['metric'] == 'auc_movement'].set_index('slice')['value']
    assert (aucs == 1.0).all()
    fps = a[a['metric'] == 'anchor_fp'].set_index('slice')['value']
    assert fps['news'] == pytest.approx(0.25)
    assert fps['news_validation'] == 0.0
    # populations pinned: 4 transfer-labeled rejects vs 4 anchors (round 1)
    npos = a[a['metric'] == 'n_pos'].set_index('slice')['value']
    assert npos['news'] == 4 and npos['news_validation'] == 3

    # b axes: strong synthetic attribute direction must be recovered
    for axis, n_slices in (('b1_paradetox', 1), ('b2_formality', 2)):
        b = card[card['axis'] == axis]
        assert len(b) == 3 * n_slices
        assert (b[b['metric'] == 'axis_auc']['value'] > 0.9).all()
        assert (b[b['metric'] == 'paired_axis_acc']['value'] > 0.9).all()
    b2 = card[card['axis'] == 'b2_formality']
    assert sorted(b2['slice'].unique()) == ['clean', 'strategy_d']


def test_score_encoder_is_deterministic(synthetic):
    enc, assets = synthetic
    a = score_encoder(enc, encoder_id='x', assets=assets, n_boot=50)
    b = score_encoder(enc, encoder_id='x', assets=assets, n_boot=50)
    pd.testing.assert_frame_equal(a.drop(columns='created_at'),
                                  b.drop(columns='created_at'))


def test_scorecard_round_trip(synthetic, tmp_path):
    enc, assets = synthetic
    card = score_encoder(enc, encoder_id='fake-v0', assets=assets,
                         axes=('b1',), n_boot=50)
    json_path, csv_path = write_scorecard(card,
                                          str(tmp_path / 'scorecard'))
    for path in (json_path, csv_path):
        back = read_scorecard(path)
        assert list(back.columns) == SCORECARD_COLUMNS
        np.testing.assert_allclose(back['value'].astype(float),
                                   card['value'].astype(float))


def test_axis_c_scoring(synthetic):
    enc, assets = synthetic
    card = score_encoder(enc, encoder_id='fake-v0', assets=assets,
                         axes=('c',), n_boot=100)
    c = card.set_index('slice')
    # synthetic authors have distinct directions -> same-author wins clearly
    for slice_ in ('pikabu_same_post', 'pikabu_cross_post',
                   'writers_reference'):
        assert c.loc[slice_, 'value'] > 0.9, slice_
    assert c.loc['pikabu_silhouette', 'value'] > 0.2
    assert c.loc['pikabu_silhouette', 'ci_low'] <= \
        c.loc['pikabu_silhouette', 'value']


def test_silhouette_by_group_basics():
    a = _unit(RNG.normal(size=DIM))
    b = _unit(RNG.normal(size=DIM))
    emb = np.stack([_unit(v + 0.01 * RNG.normal(size=DIM))
                    for v in [a, a, a, b, b, b, RNG.normal(size=DIM)]])
    groups = ['ga', 'ga', 'ga', 'gb', 'gb', 'gb', 'singleton']
    sil, kept = silhouette_by_group(emb, groups)
    assert len(sil) == 6  # singleton group excluded
    assert sil.mean() > 0.5


def test_axis_a_informal_scoring(synthetic):
    enc, assets = synthetic
    card = score_encoder(enc, encoder_id='fake-v0', assets=assets,
                         axes=('a_informal',), n_boot=100)
    assert sorted(card['slice'].unique()) == ['ficbook', 'pikabu',
                                              'taiga_proza']
    assert len(card) == 15  # 3 sources x 5 metric rows
    aucs = card[card['metric'] == 'auc_movement'].set_index('slice')['value']
    assert (aucs == 1.0).all()  # fully separable synthetic geometry
    fps = card[card['metric'] == 'anchor_fp'].set_index('slice')['value']
    assert np.allclose(fps, 0.25)


def test_unknown_axis_raises(synthetic):
    enc, assets = synthetic
    with pytest.raises(KeyError):
        score_encoder(enc, encoder_id='x', assets=assets, axes=('nope',))


def test_resolve_embed_fn_contract():
    class WithEncode:
        def encode(self, texts):
            return np.ones((len(texts), 3))

    assert resolve_embed_fn(WithEncode())(['a', 'b']).shape == (2, 3)
    assert resolve_embed_fn(lambda t: np.zeros((len(t), 2)))(['a']).shape \
        == (1, 2)
    with pytest.raises(TypeError):
        resolve_embed_fn(42)
