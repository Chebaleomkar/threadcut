import pytest
from schedule.timeparse import to_minutes
from schedule.meetings import Meeting, conflicts


@pytest.mark.parametrize("s,m", [("9:05", 545), ("09:05", 545), ("9am", 540), ("12pm", 720),
                                 ("12am", 0), ("7:30pm", 1170)])
def test_to_minutes(s, m):
    assert to_minutes(s) == m


def test_invalid():
    with pytest.raises(ValueError):
        to_minutes("25:00")


def test_back_to_back_ok():
    a, b = Meeting.parse("a", "9am", "10am"), Meeting.parse("b", "10am", "11am")
    assert conflicts([a, b]) == []


def test_conflicts_order():
    ms = [Meeting.parse("late", "2pm", "3pm"), Meeting.parse("x", "9am", "11am"),
          Meeting.parse("y", "10:30am", "12pm"), Meeting.parse("z", "2:30pm", "4pm")]
    assert conflicts(ms) == [("x", "y"), ("late", "z")]
