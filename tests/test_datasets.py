"""Offline tests for normalized benchmark dataset adapters."""

from dejavuu.eval.datasets import cases_from_speed_rows


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
