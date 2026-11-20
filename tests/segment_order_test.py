"""Transcript order is admitted consistently before any publication."""

import json

import pytest

from podcast_transcript import single_track
from podcast_transcript.transcript_validation import (
    validate_segments,
    load_whisper_transcript,
)
from single_track_test import RecordingBackend


def raw(starts):
    return [
        {"start": start, "end": start + 10, "text": f"synthetic {i}"}
        for i, start in enumerate(starts)
    ]


def dote(starts):
    # Build directly so raw admission does not mask cached DOTe admission.
    return {
        "lines": [
            {
                "startTime": f"00:00:{start:02},000",
                "endTime": f"00:00:{start + 10:02},000",
                "speakerDesignation": str(i),
                "text": f"synthetic {i}",
            }
            for i, start in enumerate(starts)
        ]
    }


def test_raw_and_cached_raw_refuse_decreasing_starts(tmp_path):
    segments = raw([5, 1])
    with pytest.raises(ValueError, match="segment 2"):
        validate_segments(segments)
    path = tmp_path / "chunk_000.json"
    path.write_text(json.dumps({"segments": segments}))
    with pytest.raises(ValueError, match="chunk_000.json.*segment 2"):
        load_whisper_transcript(path)


@pytest.mark.parametrize(
    "converter",
    [
        single_track.convert_to_webvtt,
        single_track.convert_dote_to_podlove,
        single_track.convert_to_plaintext,
    ],
)
def test_dote_export_refuses_decreasing_starts_before_write(tmp_path, converter):
    source, target = tmp_path / "cached.dote.json", tmp_path / "export"
    source.write_text(json.dumps(dote([5, 1])))
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="line 2"):
        converter(source, target)
    assert target.read_bytes() == b"previous"


@pytest.mark.parametrize("starts", [[1, 1, 2], [1, 2, 3], []])
def test_equal_starts_and_overlaps_keep_text_and_speaker_order(tmp_path, starts):
    validate_segments(raw(starts))
    source, target = tmp_path / "source.dote.json", tmp_path / "out.json"
    source.write_text(json.dumps(dote(starts)))
    single_track.convert_dote_to_podlove(source, target)
    rows = json.loads(target.read_text())["transcripts"]
    assert [row["speaker"] for row in rows] == [str(i) for i in range(len(starts))]
    assert [row["text"] for row in rows] == [
        f"synthetic {i}" for i in range(len(starts))
    ]


def test_combined_chunk_starts_cannot_go_backwards(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps(dote([5])))
    second.write_text(json.dumps(dote([1])))
    target = tmp_path / "combined.json"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="line 2"):
        single_track.combine_dote_chunks([first, second], target, [1, 1])
    assert target.read_bytes() == b"previous"


def test_invalid_backend_order_preserves_previous_generation(episode, monkeypatch):
    store, previous = episode
    backend = RecordingBackend("new", "unused")

    def infer(chunk, target):
        target.write_text(json.dumps({"segments": raw([5, 1])}))

    monkeypatch.setattr(backend, "transcribe", infer)
    with pytest.raises(ValueError, match="segment 2"):
        single_track.transcribe(store.url, backend)
    assert all(
        (store.podcast_dir / p).read_bytes() == value for p, value in previous.items()
    )


def test_raw_chunk_with_unpaired_surrogate_is_refused_before_dote(tmp_path):
    segments = raw([0, 1])
    segments[1]["text"] = "synthetic\ud800text"
    with pytest.raises(ValueError, match=r"^Invalid transcript segment 2: .*scalar"):
        validate_segments(segments)
    chunk = tmp_path / "chunk_000.json"
    chunk.write_text(json.dumps({"segments": segments}))
    with pytest.raises(
        ValueError, match=r"^chunk_000\.json: Invalid transcript segment 2: "
    ) as error:
        single_track.whisper_text_chunks_to_dote([chunk])
    assert "synthetic" not in str(error.value)
    assert not chunk.with_suffix(".dote.json").exists()


def test_raw_segment_text_keeps_paired_non_bmp_characters():
    segments = raw([0])
    segments[0]["text"] = "hello \U0001f30d"
    validate_segments(segments)
