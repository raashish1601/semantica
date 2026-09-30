"""Tests for the phase-1 benchmark harness (``semantica.benchmarks``).

Everything here must run offline: the harness ships one hand-written sample
dataset and every other loader is exercised against small inline temp files, so
the suite never touches the network or a vendor SDK.
"""
import json

import pytest

from semantica.benchmarks import (
    CORPUS,
    PER_CASE,
    get_system,
    list_datasets,
    list_systems,
    load_dataset,
    run_benchmark,
)
from semantica.benchmarks import text as bt
from semantica.benchmarks.datasets import get_loader
from semantica.benchmarks.systems import SystemUnavailable
from semantica.benchmarks.types import BenchmarkCase, BenchmarkReport, Dataset


# --------------------------------------------------------------------------- #
# registry + sample dataset
# --------------------------------------------------------------------------- #
class TestRegistry:
    def test_known_datasets_are_registered(self):
        names = list_datasets()
        for name in ("sample", "hotpotqa", "musique", "locomo"):
            assert name in names

    def test_unknown_dataset_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_loader("does-not-exist")

    def test_load_dataset_dispatches_to_loader(self):
        dataset = load_dataset("sample")
        assert dataset.name == "sample"
        assert dataset.scope == PER_CASE


class TestSampleDataset:
    def test_has_cases_with_gold_answers(self):
        dataset = load_dataset("sample")
        assert len(dataset) == 6
        for case in dataset.cases:
            assert case.question
            assert case.answers
            assert case.context

    def test_limit_truncates(self):
        assert len(load_dataset("sample", limit=2)) == 2

    def test_is_per_case_scoped(self):
        assert load_dataset("sample").scope == PER_CASE


# --------------------------------------------------------------------------- #
# file-backed loaders
# --------------------------------------------------------------------------- #
class TestHotPotQALoader:
    def _write(self, tmp_path, payload):
        path = tmp_path / "hotpot.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_official_context_shape(self, tmp_path):
        payload = [
            {
                "_id": "q1",
                "question": "Which city?",
                "answer": "Paris",
                "context": [["France", ["Paris is the capital.", "It is on the Seine."]]],
            }
        ]
        dataset = load_dataset("hotpotqa", path=self._write(tmp_path, payload))
        assert len(dataset) == 1
        case = dataset.cases[0]
        assert case.case_id == "q1"
        assert case.answers == ["Paris"]
        assert case.context == ["France: Paris is the capital. It is on the Seine."]

    def test_huggingface_context_shape(self, tmp_path):
        payload = [
            {
                "id": "q2",
                "question": "Which city?",
                "answer": "Paris",
                "context": {"title": ["France"], "sentences": [["Paris is the capital."]]},
            }
        ]
        dataset = load_dataset("hotpotqa", path=self._write(tmp_path, payload))
        assert dataset.cases[0].context == ["France: Paris is the capital."]

    def test_records_without_answer_are_skipped(self, tmp_path):
        payload = [
            {"_id": "ok", "question": "q", "answer": "a", "context": []},
            {"_id": "bad", "question": "q", "answer": "", "context": []},
        ]
        dataset = load_dataset("hotpotqa", path=self._write(tmp_path, payload))
        assert [c.case_id for c in dataset.cases] == ["ok"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset("hotpotqa", path=str(tmp_path / "nope.json"))


class TestMuSiQueLoader:
    def _write(self, tmp_path, records):
        path = tmp_path / "musique.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return str(path)

    def test_parses_jsonl_and_folds_aliases(self, tmp_path):
        records = [
            {
                "id": "m1",
                "question": "Who?",
                "answer": "Bach",
                "answer_aliases": ["Johann Sebastian Bach"],
                "paragraphs": [{"title": "Composer", "paragraph_text": "Bach wrote it."}],
            }
        ]
        dataset = load_dataset("musique", path=self._write(tmp_path, records))
        case = dataset.cases[0]
        assert case.answers == ["Bach", "Johann Sebastian Bach"]
        assert case.context == ["Composer: Bach wrote it."]

    def test_records_without_answer_are_skipped(self, tmp_path):
        records = [
            {"id": "ok", "question": "q", "answer": "a", "paragraphs": []},
            {"id": "bad", "question": "q", "answer": "", "paragraphs": []},
        ]
        dataset = load_dataset("musique", path=self._write(tmp_path, records))
        assert [c.case_id for c in dataset.cases] == ["ok"]


class TestLoCoMoLoader:
    def _write(self, tmp_path, payload):
        path = tmp_path / "locomo.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _payload(self):
        return [
            {
                "sample_id": "conv-1",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1_date_time": "1 Jan 2024",
                    "session_1": [
                        {"speaker": "Alice", "text": "I love hiking."},
                        {"speaker": "Bob", "text": "I prefer chess."},
                    ],
                    "session_2_date_time": "2 Jan 2024",
                    "session_2": [{"speaker": "Alice", "text": "I went hiking again."}],
                },
                "qa": [
                    {"question": "What does Alice love?", "answer": "hiking", "category": 4},
                    {"question": "Who likes chess?", "answer": "Bob", "category": 2},
                ],
            }
        ]

    def test_scope_is_corpus(self, tmp_path):
        dataset = load_dataset("locomo", path=self._write(tmp_path, self._payload()))
        assert dataset.scope == CORPUS

    def test_sessions_are_flattened_with_dates_and_speakers(self, tmp_path):
        dataset = load_dataset("locomo", path=self._write(tmp_path, self._payload()))
        context = dataset.cases[0].context
        assert "[1 Jan 2024]" in context
        assert "Alice: I love hiking." in context
        assert "[2 Jan 2024]" in context

    def test_category_filter(self, tmp_path):
        path = self._write(tmp_path, self._payload())
        dataset = load_dataset("locomo", path=path, categories=[4])
        assert len(dataset) == 1
        assert dataset.cases[0].metadata["category_name"] == "single-hop"

    def test_license_surfaced_as_non_commercial(self, tmp_path):
        dataset = load_dataset("locomo", path=self._write(tmp_path, self._payload()))
        assert "NON-COMMERCIAL" in dataset.license


# --------------------------------------------------------------------------- #
# text helpers
# --------------------------------------------------------------------------- #
class TestTextHelpers:
    def test_tokenize_lowercases_and_drops_punctuation(self):
        assert bt.tokenize("The Quick, brown FOX!") == ["the", "quick", "brown", "fox"]

    def test_split_sentences(self):
        assert bt.split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_overlap_score_is_share_of_question_words(self):
        assert bt.overlap_score("red apple", "the apple is red") == pytest.approx(1.0)
        assert bt.overlap_score("red apple", "blue car") == 0.0

    def test_best_sentence_picks_highest_overlap(self):
        passages = ["The capital of France is Paris.", "Paris is a large city."]
        assert bt.best_sentence("What is the capital of France?", passages) == (
            "The capital of France is Paris."
        )

    def test_best_sentence_empty_input(self):
        assert bt.best_sentence("q", []) == ""

    def test_bm25_returns_source_texts_not_tokens(self):
        index = bt.BM25()
        index.add(["alpha beta gamma", "delta epsilon"])
        hits = index.search("alpha")
        assert hits == ["alpha beta gamma"]

    def test_bm25_ranks_best_match_first(self):
        index = bt.BM25()
        index.add(["unrelated text here", "alpha beta gamma alpha"])
        assert index.search("alpha beta", top_k=1) == ["alpha beta gamma alpha"]

    def test_bm25_empty_index_returns_empty(self):
        assert bt.BM25().search("anything") == []

    def test_bm25_clear_empties_index(self):
        index = bt.BM25()
        index.add(["alpha"])
        index.clear()
        assert len(index) == 0
        assert index.search("alpha") == []


# --------------------------------------------------------------------------- #
# systems
# --------------------------------------------------------------------------- #
class TestSystemRegistry:
    def test_reference_systems_are_listed(self):
        names = list_systems()
        assert "lexical" in names
        assert "semantica" in names

    def test_unknown_system_raises_unavailable(self):
        with pytest.raises(SystemUnavailable):
            get_system("no-such-system")


class TestLexicalSystem:
    def test_extracts_the_answer_sentence(self):
        system = get_system("lexical")
        system.reset()
        system.ingest(
            [
                "The telephone inventor Alexander Graham Bell was born in Scotland.",
                "The telephone was patented in the United States.",
            ],
            case_id="c1",
        )
        answer = system.answer("Which country was the telephone inventor born in?", case_id="c1")
        assert answer == "The telephone inventor Alexander Graham Bell was born in Scotland."

    def test_reset_clears_memory(self):
        system = get_system("lexical")
        system.ingest(["only passage"], case_id="c1")
        system.reset()
        assert system.answer("only passage?", case_id="c1") == ""

    def test_answer_records_retrieved_passages(self):
        system = get_system("lexical")
        system.reset()
        system.ingest(["alpha bravo", "charlie delta"], case_id="c1")
        system.answer("alpha bravo?", case_id="c1")
        assert system.last_retrieved == ["alpha bravo"]

    def test_reset_clears_retrieved_passages(self):
        system = get_system("lexical")
        system.ingest(["alpha"], case_id="c1")
        system.answer("alpha", case_id="c1")
        system.reset()
        assert system.last_retrieved == []


# --------------------------------------------------------------------------- #
# end-to-end (offline)
# --------------------------------------------------------------------------- #
class TestRunBenchmark:
    def test_sample_plus_lexical_scores_above_floor(self):
        report = run_benchmark([load_dataset("sample")], ["lexical"])
        assert isinstance(report, BenchmarkReport)
        assert report.scores["lexical/sample"] > 0.0
        result = report.results[0]
        assert result.n == 6
        assert result.errors == 0

    def test_unavailable_system_is_skipped_not_fatal(self):
        report = run_benchmark([load_dataset("sample")], ["no-such-system"])
        assert report.results == []
        assert [entry["system"] for entry in report.skipped] == ["no-such-system"]

    def test_strict_mode_raises_on_unavailable_system(self):
        with pytest.raises(SystemUnavailable):
            run_benchmark([load_dataset("sample")], ["no-such-system"], on_error="raise")

    def test_markdown_table_mentions_system_and_dataset(self):
        report = run_benchmark([load_dataset("sample")], ["lexical"])
        table = report.to_markdown_table()
        assert "lexical" in table
        assert "sample" in table

    def test_report_serializes(self):
        report = run_benchmark([load_dataset("sample")], ["lexical"])
        payload = report.as_dict()
        assert payload["systems"] == ["lexical"]
        assert payload["datasets"] == ["sample"]
        assert "skipped" in payload

    def test_corpus_scope_ingests_once(self):
        dataset = Dataset(
            name="mini",
            license="test",
            scope=CORPUS,
            cases=[
                BenchmarkCase(case_id="a", question="Where is Paris?",
                              answers=["France"], context=["Paris is in France."]),
                BenchmarkCase(case_id="b", question="Where is Paris?",
                              answers=["France"], context=["Paris is in France."]),
            ],
        )
        report = run_benchmark([dataset], ["lexical"])
        assert report.results[0].n == 2
        assert report.results[0].errors == 0

    def test_predictions_carry_retrieved_passages(self):
        report = run_benchmark([load_dataset("sample")], ["lexical"])
        predictions = report.results[0].predictions
        assert predictions
        # Every sample question shares content words with its evidence, so the
        # lexical floor always retrieves something; the report must show it,
        # otherwise per-case retrieval (the main debugging surface) is invisible.
        assert all(prediction.retrieved for prediction in predictions)
        by_id = {prediction.case_id: prediction for prediction in predictions}
        assert by_id["sample-000"].retrieved[0] == (
            "The novel Neuromancer was written by William Gibson and published in 1984."
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class TestCli:
    def test_list_command_exits_cleanly(self, capsys):
        from semantica.benchmarks.__main__ import main

        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "sample" in out

    def test_run_command_offline(self, capsys):
        from semantica.benchmarks.__main__ import main

        code = main(["run", "--dataset", "sample", "--system", "lexical", "--quiet"])
        assert code == 0
        assert "lexical" in capsys.readouterr().out
