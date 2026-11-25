"""Synthetic native tool admission tests."""

import errno
import json
import subprocess
import sys
import time
from io import BytesIO
from types import SimpleNamespace

import pytest

from podcast_transcript import audio, backends, safe_io, single_track


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_probe_capture_overflow_reaps_child(monkeypatch, tmp_path, stream):
    popen = subprocess.Popen
    children = []

    # The baseline uses subprocess.run; both paths go through Popen.
    def launch(cmd, **kw):
        script = f"import sys,time; sys.{stream}.buffer.write(b'x'*1000000); sys.{stream}.flush(); time.sleep(30)"
        child = popen([sys.executable, "-c", script], **kw)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(audio, "FFPROBE_TIMEOUT_SECONDS", 10)
    store = audio.Audio(base_dir=tmp_path, url="https://example.test/audio.mp3")
    store.episode_chunks_dir.mkdir(parents=True)
    chunks = [store.episode_chunks_dir / f"chunk_{i:03d}.mp3" for i in range(2)]
    paths = chunks + [
        store.cache_metadata_path,
        store.chunks_manifest_path,
        store.output_path(".txt"),
    ]
    for path in paths:
        path.write_bytes(b"previous")
    store.cache_metadata_path.write_text('{"backend":"previous"}')
    store.chunks_manifest_path.write_text('["chunk_000.mp3","chunk_001.mp3"]')
    before = {path: path.read_bytes() for path in paths}
    monkeypatch.setattr(store, "prepare_audio_for_transcription", lambda: chunks)
    backend = backends.WhisperCpp()
    monkeypatch.setattr(
        backend,
        "transcribe",
        lambda *args: pytest.fail("invalid audio reached inference"),
    )
    with pytest.raises(RuntimeError, match="output.*limit") as error:
        single_track._transcribe_locked(store, backend)
    assert len(str(error.value)) < 2500
    assert children and children[0].poll() is not None
    assert all(path.read_bytes() == before[path] for path in paths)


def native(timestamp, *, seek=0):
    return {
        "transcription": [
            {
                "offsets": {"from": seek},
                "timestamps": {"from": timestamp, "to": timestamp},
                "text": "private-synthetic-text",
            }
        ]
    }


@pytest.mark.parametrize(
    "timestamp",
    [
        "00:99:00,000",
        "00:00:99,000",
        "-1:00:00,000",
        "00:00:00,1",
        "00:00:00,1234",
        "00:00:00.000",
        "0:00:00,000",
        None,
    ],
)
def test_native_invalid_timestamp_preserves_output(tmp_path, timestamp):
    source = tmp_path / "native.json"
    source.write_text(json.dumps(native(timestamp)))
    target = tmp_path / "transcript.json"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="Invalid native transcript") as error:
        backends.WhisperCpp().convert_output_format(source, target)
    assert "private-synthetic-text" not in str(error.value)
    assert target.read_bytes() == b"previous"


@pytest.mark.parametrize("seek", [True, 1.5, 1.0, "1", -1])
def test_native_invalid_seek_preserves_output(tmp_path, seek):
    source = tmp_path / "native.json"
    source.write_text(json.dumps(native("00:00:00,000", seek=seek)))
    target = tmp_path / "transcript.json"
    target.write_bytes(b"previous")

    with pytest.raises(
        ValueError, match="Invalid native transcript.*integer seek"
    ) as error:
        backends.WhisperCpp().convert_output_format(source, target)

    assert "private-synthetic-text" not in str(error.value)
    assert target.read_bytes() == b"previous"


@pytest.mark.parametrize("seek", [0, 7_199_000])
def test_native_integer_seek_is_preserved(tmp_path, seek):
    source = tmp_path / "native.json"
    source.write_text(json.dumps(native("01:59:59,000", seek=seek)))
    target = tmp_path / "transcript.json"

    backends.WhisperCpp().convert_output_format(source, target)

    actual_seek = json.loads(target.read_text())["segments"][0]["seek"]
    assert type(actual_seek) is int
    assert actual_seek == seek


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"transcription": None},
        {"transcription": [{}]},
        {"transcription": [None]},
        {"transcription": [{"text": "private-synthetic-text"}]},
    ],
)
def test_native_missing_schema_is_normalized(tmp_path, data):
    source = tmp_path / "native.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid native transcript"):
        backends.WhisperCpp().convert_output_format(source, tmp_path / "out.json")


def test_native_multiday_timecode(tmp_path):
    result = backends.WhisperCpp.transform_transcription(native("123:59:59,999"))
    assert result["segments"][0]["start"] == pytest.approx(446399.999)


def test_probe_small_success_failure_and_timeout(monkeypatch, tmp_path):
    popen = subprocess.Popen
    programs = iter(
        [
            'print(\'{"format": {"duration": "2.5"}}\')',
            "import sys; sys.stderr.write('synthetic failure'); sys.exit(7)",
            "import time; time.sleep(30)",
        ]
    )
    children = []

    def launch(cmd, **kw):
        child = popen([sys.executable, "-c", next(programs)], **kw)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(audio, "FFPROBE_TIMEOUT_SECONDS", 10)
    assert audio.get_audio_duration(tmp_path / "audio.mp3") == 2.5
    with pytest.raises(RuntimeError, match="exit code 7.*synthetic failure"):
        audio.get_audio_duration(tmp_path / "audio.mp3")
    monkeypatch.setattr(audio, "FFPROBE_TIMEOUT_SECONDS", 0.5)
    with pytest.raises(RuntimeError, match="timed out"):
        audio.get_audio_duration(tmp_path / "audio.mp3")
    assert all(child.poll() is not None for child in children)


def test_failure_signals_group_before_reaping_leader(monkeypatch):
    cleanup = safe_io._kill_owned_tree
    calls = []

    def retire_unreaped(process):
        assert process.returncode is None
        calls.append(process)
        cleanup(process)

    monkeypatch.setattr(safe_io, "_kill_owned_tree", retire_unreaped)
    with pytest.raises(RuntimeError, match="exit code 7"):
        safe_io.capture_checked(
            [sys.executable, "-c", "import sys;sys.exit(7)"],
            description="synthetic",
            timeout=10,
        )
    assert len(calls) == 1
    assert calls[0].returncode == 7


@pytest.mark.parametrize("startup, exit_delay", [(10, 0), (6, 4), (6, 3)])
def test_capture_shares_deadline_between_startup_and_exit(mocker, startup, exit_delay):
    now = 100
    process = mocker.MagicMock(pid=123, returncode=None)
    process.__enter__.return_value = process
    process.stdout = BytesIO(b"complete")
    process.stderr = BytesIO()

    def launch(*args, **kwargs):
        nonlocal now
        now += startup
        return process

    def observe_exit(*args):
        nonlocal now
        now += exit_delay
        return object()

    def retire(owned):
        owned.returncode = 0

    mocker.patch.object(time, "monotonic", side_effect=lambda: now)
    mocker.patch.object(subprocess, "Popen", side_effect=launch)
    mocker.patch.object(
        safe_io,
        "os",
        SimpleNamespace(
            name="posix", waitid=observe_exit, P_PID=1, WEXITED=2, WNOWAIT=4, WNOHANG=8
        ),
    )
    cleanup = mocker.patch.object(safe_io, "_kill_owned_tree", side_effect=retire)

    cmd = ["synthetic-probe"]
    if startup + exit_delay >= 10:
        with pytest.raises(RuntimeError, match="synthetic timed out after 10 seconds"):
            safe_io.capture_checked(cmd, description="synthetic", timeout=10)
    else:
        assert (
            safe_io.capture_checked(cmd, description="synthetic", timeout=10)
            == b"complete"
        )
    cleanup.assert_called_once_with(process)


@pytest.mark.parametrize(
    "observation", ["exit", "registration-event", "registration-error", "late-esrch"]
)
def test_capture_requires_owned_kqueue_exit_observation(mocker, observation):
    process = mocker.MagicMock(pid=123, returncode=None, stdout=None, stderr=None)
    process.__enter__.return_value = process
    mocker.patch.object(subprocess, "Popen", return_value=process)
    mocker.patch.object(safe_io, "os", SimpleNamespace(name="posix"))
    clock = mocker.patch.object(time, "monotonic", return_value=0.0)
    queue = mocker.Mock()
    registration = object()
    kqueue = SimpleNamespace(
        kqueue=lambda: queue,
        kevent=mocker.Mock(return_value=registration),
        KQ_FILTER_PROC=-5,
        KQ_EV_ADD=1,
        KQ_EV_ONESHOT=16,
        KQ_EV_ERROR=16384,
        KQ_NOTE_EXIT=1 << 31,
    )
    mocker.patch.object(safe_io, "select", kqueue)

    def event(*, pid=123, filter_id=kqueue.KQ_FILTER_PROC, flags=0, fflags=0, data=0):
        return SimpleNamespace(
            ident=pid, filter=filter_id, flags=flags, fflags=fflags, data=data
        )

    esrch = event(flags=kqueue.KQ_EV_ERROR, data=errno.ESRCH)
    if observation == "exit":
        results = iter(
            [
                [event()],
                [event(pid=456, fflags=kqueue.KQ_NOTE_EXIT)],
                [event(filter_id=-1, fflags=kqueue.KQ_NOTE_EXIT)],
                [event(fflags=kqueue.KQ_NOTE_EXIT)],
            ]
        )
    elif observation == "registration-event":
        results = iter([[esrch]])
    elif observation == "registration-error":
        results = iter([ProcessLookupError(errno.ESRCH, "synthetic")])
    else:
        results = iter([[], [esrch]])

    def control(changes, max_events, timeout):
        clock.return_value += 0.02
        result = next(results)
        if isinstance(result, OSError):
            raise result
        return result

    def retire(owned):
        assert owned.returncode is None
        owned.returncode = 0

    queue.control.side_effect = control
    cleanup = mocker.patch.object(safe_io, "_kill_owned_tree", side_effect=retire)
    cmd = ["synthetic-probe"]
    if observation == "late-esrch":
        with pytest.raises(RuntimeError, match="Could not observe owned command exit"):
            safe_io.capture_checked(cmd, description="synthetic", timeout=0.09)
    else:
        assert (
            safe_io.capture_checked(cmd, description="synthetic", timeout=0.09) == b""
        )
    waits = [call.args[2] for call in queue.control.call_args_list]
    expected_waits = {
        "exit": [0.05, 0.05, 0.05, 0.03],
        "registration-event": [0.05],
        "registration-error": [0.05],
        "late-esrch": [0.05, 0.05],
    }
    assert waits == pytest.approx(expected_waits[observation])
    assert [call.args[0] for call in queue.control.call_args_list] == [
        [registration],
        *([[]] * (len(waits) - 1)),
    ]
    queue.close.assert_called_once_with()
    cleanup.assert_called_once_with(process)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_probe_capture_requires_finite_timeout(timeout):
    with pytest.raises(ValueError, match="positive finite"):
        safe_io.capture_checked(
            ["never-started"], description="synthetic", timeout=timeout
        )


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "8"])
def test_capture_limits_are_validated_before_launch(mocker, stream, limit):
    launch = mocker.patch.object(
        subprocess, "Popen", side_effect=AssertionError("unexpected process launch")
    )
    with pytest.raises(ValueError, match=rf"{stream} limit.*positive integer"):
        safe_io.capture_checked(
            ["never-started"],
            description="synthetic",
            timeout=10,
            stderr_tail=True,
            **{f"{stream}_limit": limit},
        )
    launch.assert_not_called()


def test_capture_discards_stdout_and_keeps_small_stderr_tail():
    with pytest.raises(RuntimeError, match=r"exit code 7: \[stderr truncated\]\ntail$"):
        safe_io.capture_checked(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 100_000); "
                "sys.stderr.write('discard this tail'); sys.exit(7)",
            ],
            description="synthetic",
            timeout=10,
            stdout_limit=None,
            stderr_limit=4,
            stderr_tail=True,
        )


@pytest.mark.parametrize("capture", [True, False])
def test_probe_parent_exit_keeps_identity_until_pipe_cleanup(monkeypatch, capture):
    import threading

    cleaned = threading.Event()

    class Pipe:
        def read(self, size):
            cleaned.wait(2)
            return b""

        def close(self):
            pass

    class ExitedLeader:
        returncode = None
        stdout = Pipe() if capture else None
        stderr = Pipe()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait(self, timeout=None):
            # Model an exited leader. Waiting would release its reserved PID.
            self.returncode = 0
            return 0

    process = ExitedLeader()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kw: process)

    def cleanup(owned):
        assert owned is process
        assert owned.returncode is None, "leader identity released before pipe cleanup"
        cleaned.set()
        owned.returncode = -1

    monkeypatch.setattr(safe_io, "_kill_owned_tree", cleanup)
    with pytest.raises(RuntimeError, match="timed out"):
        run = safe_io.capture_checked if capture else safe_io.run_checked
        run(["synthetic-probe"], description="synthetic probe", timeout=0.05)
    assert cleaned.is_set()
