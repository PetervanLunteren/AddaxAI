"""Machine facts are diagnostics-only: a sane value on the platforms CI
runs on, and never an exception, whatever the machine answers."""

from unittest.mock import patch

from app.utils import machine_info


def test_total_memory_bytes_is_a_plausible_ram_size():
    value = machine_info.total_memory_bytes()
    assert isinstance(value, int)
    # A machine running this suite has over 256 MB and under 64 TB.
    assert 256 * 1024**2 < value < 64 * 1024**4


def test_cpu_name_is_a_non_empty_string():
    value = machine_info.cpu_name()
    assert isinstance(value, str)
    assert value


def test_a_machine_that_cannot_answer_gives_none_not_an_error():
    with (
        patch.object(machine_info.sys, "platform", "darwin"),
        patch.object(machine_info, "_run", side_effect=OSError("no sysctl")),
    ):
        assert machine_info.total_memory_bytes() is None
        assert machine_info.cpu_name() is None
