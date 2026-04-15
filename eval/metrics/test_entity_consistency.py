"""
Unit tests for EntityConsistencyScorer (Task 1.7).

Run on remote:
    ssh tallin.vpn 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate tst311 && \
        cd /home/abragin/src/textprism/ && \
        pytest tst_utils/eval/metrics/test_entity_consistency.py -v'
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture: single scorer instance reused across tests (NER model is slow to load)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scorer():
    from tst_utils.eval.metrics.entity_consistency import EntityConsistencyScorer
    return EntityConsistencyScorer(device="cuda", pre_clean=False, use_labse=False)


@pytest.fixture(scope="module")
def scorer_labse():
    from tst_utils.eval.metrics.entity_consistency import EntityConsistencyScorer
    return EntityConsistencyScorer(device="cuda", pre_clean=False, use_labse=True)


@pytest.fixture(scope="module")
def scorer_d():
    from tst_utils.eval.metrics.entity_consistency import EntityConsistencyScorer
    return EntityConsistencyScorer(device="cuda", pre_clean=False, use_bert_score=True)


# ---------------------------------------------------------------------------
# score_pair: return type and field presence
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_keys_present(self, scorer):
        r = scorer.score_pair("Жилин пошёл домой.", "Маруся пошла домой.")
        assert "entity_score" in r
        assert "has_substitution" in r
        assert "new_entities" in r
        assert "src_entities" in r
        assert "tgt_entities" in r

    def test_score_range(self, scorer):
        r = scorer.score_pair("Наташа пела.", "Маргарита пела.")
        assert 0.0 <= r["entity_score"] <= 1.0

    def test_labse_key_present_when_enabled(self, scorer_labse):
        r = scorer_labse.score_pair("Иван пошёл.", "Маруся пошла.")
        assert "entity_bag_sim" in r
        assert 0.0 <= r["entity_bag_sim"] <= 1.0

    def test_labse_key_absent_when_disabled(self, scorer):
        r = scorer.score_pair("Иван пошёл.", "Маруся пошла.")
        assert "entity_bag_sim" not in r


# ---------------------------------------------------------------------------
# score_pair: clear substitution
# ---------------------------------------------------------------------------

class TestClearSubstitution:
    def test_person_substitution_flagged(self, scorer):
        """Injected person name not in source → has_substitution=True."""
        r = scorer.score_pair(
            "Жилин не понял старика.",
            "Маруся не поняла старика.",
        )
        assert r["has_substitution"] is True
        assert r["entity_score"] < 1.0
        assert len(r["new_entities"]) >= 1

    def test_person_injection_into_entityless_source(self, scorer):
        """Target has a name, source has none → flagged."""
        r = scorer.score_pair(
            "Они сидели молча.",
            "Маруся сидела молча.",
        )
        assert r["has_substitution"] is True

    def test_entity_score_zero_when_all_new(self, scorer):
        """All target entities are new → score near 0."""
        r = scorer.score_pair(
            "Они сидели молча.",
            "Наташа и Андрей сидели молча.",
        )
        assert r["entity_score"] == pytest.approx(0.0)

    def test_new_entities_list_nonempty(self, scorer):
        r = scorer.score_pair(
            "Жилин не понял старика.",
            "Маруся не поняла старика.",
        )
        assert isinstance(r["new_entities"], list)
        assert len(r["new_entities"]) > 0
        # Each entry is (entity_type, entity_word)
        etype, eword = r["new_entities"][0]
        assert isinstance(etype, str)
        assert isinstance(eword, str)


# ---------------------------------------------------------------------------
# score_pair: morphological variants (should NOT be flagged)
# ---------------------------------------------------------------------------

class TestMorphologicalVariants:
    def test_city_inflection_not_flagged(self, scorer):
        """Ленинград → Ленинграде is a morphological variant, not a substitution."""
        r = scorer.score_pair(
            "Он жил в Ленинграде.",
            "Он родился в Ленинграду.",
        )
        assert r["has_substitution"] is False

    def test_person_name_inflection_not_flagged(self, scorer):
        """Иван → Ивану is a case variant, not a substitution."""
        r = scorer.score_pair(
            "Я написал Ивану письмо.",
            "Она написала письмо Ивану.",
        )
        assert r["has_substitution"] is False

    def test_identical_pair(self, scorer):
        """Identical source and target → perfect score."""
        text = "Наташа Ростова любила танцевать в Москве."
        r = scorer.score_pair(text, text)
        assert r["entity_score"] == pytest.approx(1.0)
        assert r["has_substitution"] is False
        assert r["new_entities"] == []


# ---------------------------------------------------------------------------
# score_pair: no entities on either side
# ---------------------------------------------------------------------------

class TestNoEntities:
    def test_no_entities_both_sides(self, scorer):
        """No named entities in either text → no substitution, score=1.0."""
        r = scorer.score_pair(
            "Они сидели на кроватях.",
            "Они неподвижно сидели.",
        )
        assert r["entity_score"] == pytest.approx(1.0)
        assert r["has_substitution"] is False
        assert r["new_entities"] == []

    def test_empty_strings(self, scorer):
        """Empty inputs → no substitution."""
        r = scorer.score_pair("", "")
        assert r["entity_score"] == pytest.approx(1.0)
        assert r["has_substitution"] is False


# ---------------------------------------------------------------------------
# score_pair: entity_bag_sim (LaBSE, Approach C)
# ---------------------------------------------------------------------------

class TestEntityBagSim:
    def test_bag_sim_high_for_same_entities(self, scorer_labse):
        r = scorer_labse.score_pair(
            "Наташа пошла в Москву.",
            "Наташу отправили в Москве.",
        )
        assert r["entity_bag_sim"] > 0.9

    def test_bag_sim_low_for_different_persons(self, scorer_labse):
        r = scorer_labse.score_pair(
            "Жилин не понял старика.",
            "Маруся не поняла старика.",
        )
        assert r["entity_bag_sim"] < 0.5

    def test_bag_sim_one_if_no_entities_both(self, scorer_labse):
        r = scorer_labse.score_pair("Они молчали.", "Они молчали тихо.")
        assert r["entity_bag_sim"] == pytest.approx(1.0)

    def test_bag_sim_zero_if_entities_appear_from_nothing(self, scorer_labse):
        r = scorer_labse.score_pair("Они молчали.", "Маруся молчала.")
        assert r["entity_bag_sim"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_batch: shape consistency and values match score_pair
# ---------------------------------------------------------------------------

class TestScoreBatch:
    SOURCES = [
        "Жилин не понял старика.",
        "Он жил в Ленинграде.",
        "Они сидели молча.",
    ]
    TARGETS = [
        "Маруся не поняла старика.",   # substitution
        "Он жил в Ленинграду.",        # morphological variant
        "Они сидели молча.",            # no change
    ]

    def test_output_shapes(self, scorer):
        out = scorer.score_batch(self.SOURCES, self.TARGETS)
        assert out["entity_score"].shape == (3,)
        assert out["has_substitution"].shape == (3,)
        assert len(out["new_entities"]) == 3

    def test_batch_matches_individual(self, scorer):
        out = scorer.score_batch(self.SOURCES, self.TARGETS)
        for i, (s, t) in enumerate(zip(self.SOURCES, self.TARGETS)):
            r = scorer.score_pair(s, t)
            assert out["entity_score"][i] == pytest.approx(r["entity_score"])
            assert out["has_substitution"][i] == r["has_substitution"]

    def test_batch_labse_key_present(self, scorer_labse):
        out = scorer_labse.score_batch(self.SOURCES, self.TARGETS)
        assert "entity_bag_sim" in out
        assert out["entity_bag_sim"].shape == (3,)

    def test_empty_batch(self, scorer):
        out = scorer.score_batch([], [])
        assert len(out["entity_score"]) == 0
        assert len(out["new_entities"]) == 0


# ---------------------------------------------------------------------------
# Approach D: NE-filtered BERTScore
# ---------------------------------------------------------------------------

class TestBertScore:
    def test_d_keys_present(self, scorer_d):
        r = scorer_d.score_pair("Жилин пошёл домой.", "Маруся пошла домой.")
        assert "ne_bert_precision" in r
        assert "ne_bert_recall" in r
        assert "ne_bert_f1" in r
        assert "combined_entity_score" in r

    def test_d_score_range(self, scorer_d):
        r = scorer_d.score_pair("Наташа пела.", "Маргарита пела.")
        assert 0.0 <= r["ne_bert_precision"] <= 1.0
        assert 0.0 <= r["ne_bert_recall"] <= 1.0
        assert 0.0 <= r["ne_bert_f1"] <= 1.0
        assert 0.0 <= r["combined_entity_score"] <= 1.0

    def test_d_identical_pair_high_precision(self, scorer_d):
        """Identical texts → target NE tokens closely match source → precision near 1."""
        text = "Наташа Ростова танцевала в Москве."
        r = scorer_d.score_pair(text, text)
        assert r["ne_bert_precision"] > 0.95

    def test_d_clear_substitution_low_precision(self, scorer_d):
        """Injected name not in source → target NE tokens have no close source match."""
        r_sub = scorer_d.score_pair("Жилин не понял старика.", "Маруся не поняла старика.")
        r_ok  = scorer_d.score_pair("Наташа пела.", "Наташе пели.")
        assert r_sub["ne_bert_precision"] < r_ok["ne_bert_precision"]

    def test_d_combined_score_is_min(self, scorer_d):
        """combined_entity_score == min(entity_score, ne_bert_precision)."""
        r = scorer_d.score_pair("Жилин не понял старика.", "Маруся не поняла старика.")
        assert r["combined_entity_score"] == pytest.approx(
            min(r["entity_score"], r["ne_bert_precision"])
        )

    def test_d_batch_matches_individual(self, scorer_d):
        sources = ["Жилин не понял старика.", "Наташа пела.", "Они молчали."]
        targets = ["Маруся не поняла старика.", "Наташе пели.", "Они молчали тихо."]
        out = scorer_d.score_batch(sources, targets)
        assert "ne_bert_precision" in out
        assert "combined_entity_score" in out
        assert out["ne_bert_precision"].shape == (3,)
        assert out["combined_entity_score"].shape == (3,)
        for i, (s, t) in enumerate(zip(sources, targets)):
            r = scorer_d.score_pair(s, t)
            assert out["ne_bert_precision"][i] == pytest.approx(r["ne_bert_precision"], abs=1e-5)
            assert out["combined_entity_score"][i] == pytest.approx(r["combined_entity_score"], abs=1e-5)


# ---------------------------------------------------------------------------
# Span merging: partial name references must not be flagged
# ---------------------------------------------------------------------------

class TestSpanMerging:
    def test_full_name_source_partial_target_not_flagged(self, scorer):
        """Source has full name (FIRST + LAST); target uses first name only in different case.
        After span merge, component words must still be available for matching."""
        r = scorer.score_pair(
            "Иван Иванов пошёл домой.",
            "Ивану сказали идти домой.",
        )
        assert r["has_substitution"] is False

    def test_full_name_inflected_not_flagged(self, scorer):
        """Full name in source (nom.) vs full name in target (gen.) — same person."""
        r = scorer.score_pair(
            "Анна Каренина вошла в комнату.",
            "Анны Карениной не было дома.",
        )
        assert r["has_substitution"] is False

    def test_different_full_name_flagged(self, scorer):
        """Different full name in target → still flagged even with span merging."""
        r = scorer.score_pair(
            "Иван Иванов пошёл домой.",
            "Пётр Петров пошёл домой.",
        )
        assert r["has_substitution"] is True
