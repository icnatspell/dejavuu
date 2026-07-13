"""Offline tests for normalized benchmark dataset adapters."""

import pytest

from dejavuu.eval.datasets import (
    ConversationCase,
    Turn,
    cases_from_speed_rows,
    load_cases,
    select_cases,
)


def test_speedbench_adapter_preserves_provenance_and_first_turn():
    cases = cases_from_speed_rows(
        [
            {
                "question_id": "speed-7",
                "category": "coding",
                "sub_category": "Python",
                "turns": ["Write a function.", "Now optimize it."],
                "source": "example-source",
                "difficulty": "hard",
                "multiturn": True,
            }
        ]
    )

    assert len(cases) == 1
    assert cases[0].case_id == "speed-7"
    assert cases[0].category == "coding"
    assert cases[0].prompt == "Write a function."
    assert cases[0].metadata == {
        "source": "example-source",
        "sub_category": "Python",
        "difficulty": "hard",
        "multiturn": True,
        "turn_count": 2,
    }


def test_speedbench_adapter_preserves_the_complete_conversation():
    cases = cases_from_speed_rows(
        [
            {
                "question_id": "speed-8",
                "category": "reasoning",
                "turns": ["First question", "Follow-up question"],
            }
        ]
    )

    assert [turn.text for turn in cases[0].turns] == ["First question", "Follow-up question"]


def test_every_dataset_uses_the_same_deterministic_per_category_selection():
    cases = [
        ConversationCase("a1", "a", (Turn("1"),)),
        ConversationCase("a2", "a", (Turn("2"),)),
        ConversationCase("b1", "b", (Turn("3"),)),
    ]

    assert [case.case_id for case in select_cases(cases, n=99, per_category=1)] == [
        "a1",
        "b1",
    ]


def test_official_speed_throughput_rejects_the_single_request_runner():
    with pytest.raises(ValueError, match="batched serving runner"):
        load_cases(
            "speedbench",
            split="throughput_128",
            revision=None,
            protocol="official",
            n=1,
            per_category=0,
        )
