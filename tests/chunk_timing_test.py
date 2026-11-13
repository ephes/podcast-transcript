"""Duration admission and an independent exact-decimal timing oracle."""

import json
import subprocess
import wave
from decimal import Decimal, ROUND_HALF_EVEN

import pytest

from podcast_transcript.audio import get_audio_duration
from podcast_transcript.single_track import combine_dote_chunks, whisper_to_dote
from podcast_transcript.transcript_validation import parse_dote_timecode


def chunk(path):
    path.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "startTime": "00:00:00,000",
                        "endTime": "00:00:00,001",
                        "text": "synthetic",
                    }
                ]
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "duration",
    [
        -1,
        True,
        False,
        float("nan"),
        float("inf"),
        -float("inf"),
        1e308,
        10**400,
        "1",
        None,
    ],
)
@pytest.mark.parametrize("count", [1, 2])
def test_all_durations_are_admitted_before_any_input_or_output_io(
    tmp_path, monkeypatch, duration, count
):
    paths = [chunk(tmp_path / f"{i}.json") for i in range(count)]
    before = [p.read_bytes() for p in paths]
    output = tmp_path / "output.json"
    output.write_bytes(b"last complete output")
    monkeypatch.setattr(
        "podcast_transcript.single_track.load_dote",
        lambda _: pytest.fail("opened chunk before duration admission"),
    )
    with pytest.raises(ValueError, match="duration"):
        combine_dote_chunks(paths, output, [0] * (count - 1) + [duration])
    assert output.read_bytes() == b"last complete output"
    assert [p.read_bytes() for p in paths] == before


@pytest.mark.parametrize("duration", [0, 0.0, 1.25])
def test_valid_single_chunk_keeps_exact_bytes(tmp_path, duration):
    source = chunk(tmp_path / "chunk.json")
    output = tmp_path / "output.json"
    combine_dote_chunks([source], output, [duration])
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    "durations",
    [[0.0006] * 101, [0.0000006] * 1001, [0.0005, 0.001, 0.001], [0.1] * 37],
)
def test_fractional_boundaries_match_decimal_oracle(tmp_path, durations):
    paths = [chunk(tmp_path / f"{i}.json") for i in range(len(durations))]
    output = tmp_path / "output.json"
    combine_dote_chunks(paths, output, durations)
    total = Decimal(0)
    for line, duration in zip(json.loads(output.read_text())["lines"], durations):
        expected = int((total * 1000).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
        assert parse_dote_timecode(line["startTime"]) == expected
        assert parse_dote_timecode(line["endTime"]) == expected + 1
        total += Decimal(str(duration))


def test_equivalent_partitions_and_raw_cues_use_same_rounding(tmp_path):
    final = []
    for durations in ([0.0006] * 5 + [0], [0.003, 0]):
        paths = [chunk(tmp_path / f"{i}.json") for i in range(len(durations))]
        output = tmp_path / "output.json"
        combine_dote_chunks(paths, output, durations)
        final.append(json.loads(output.read_text())["lines"][-1]["startTime"])
    raw = whisper_to_dote([{"start": 0.003, "end": 0.003, "text": "synthetic"}])
    assert final == [raw["lines"][0]["startTime"]] * 2


def test_real_tiny_audio_fractional_duration(tmp_path):
    source = tmp_path / "tiny.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 123)  # 7.6875 ms of generated silence
    subprocess.run(["ffprobe", "-version"], check=True, capture_output=True)
    duration = get_audio_duration(source)
    paths = [chunk(tmp_path / f"{i}.json") for i in range(2)]
    output = tmp_path / "output.json"
    combine_dote_chunks(paths, output, [duration, duration])
    assert json.loads(output.read_text())["lines"][1]["startTime"] == "00:00:00,008"
