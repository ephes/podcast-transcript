"""Aggregate episode admission, with synthetic chunks and small injected limits."""

import json

import pytest

from podcast_transcript import single_track
from single_track_test import RecordingBackend


def chunk(path, text="é"):
    data = {
        "lines": [
            {"startTime": "00:00:00,000", "endTime": "00:00:00,001", "text": text}
        ]
    }
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("limit", ["lines", "bytes"])
def test_many_small_chunks_refuse_before_publication(tmp_path, monkeypatch, limit):
    chunks = [chunk(tmp_path / f"{i}.json") for i in range(4)]
    output = tmp_path / "combined.json"
    output.write_bytes(b"previous")
    monkeypatch.setattr(
        single_track,
        "COMBINED_MAX_LINES",
        3 if limit == "lines" else 100,
        raising=False,
    )
    monkeypatch.setattr(
        single_track,
        "COMBINED_MAX_BYTES",
        300 if limit == "bytes" else 10000,
        raising=False,
    )
    with pytest.raises(ValueError, match="Combined transcript.*limit"):
        single_track.combine_dote_chunks(chunks, output, [1] * 4)
    assert output.read_bytes() == b"previous"


def test_exact_episode_limits_preserve_bytes_and_timestamps(tmp_path, monkeypatch):
    paths = [chunk(tmp_path / f"{i}.json") for i in range(3)]
    expected = {
        "lines": [
            dict(
                json.loads(p.read_text())["lines"][0],
                startTime=f"00:00:0{i},000",
                endTime=f"00:00:0{i},001",
            )
            for i, p in enumerate(paths)
        ]
    }
    expected_bytes = json.dumps(expected).encode("utf-8")
    monkeypatch.setattr(single_track, "COMBINED_MAX_LINES", 3, raising=False)
    monkeypatch.setattr(
        single_track, "COMBINED_MAX_BYTES", len(expected_bytes), raising=False
    )
    output = tmp_path / "combined.json"
    single_track.combine_dote_chunks(paths, output, [1] * 3)
    assert output.read_bytes() == expected_bytes
    monkeypatch.setattr(single_track, "COMBINED_MAX_BYTES", len(expected_bytes) - 1)
    with pytest.raises(ValueError, match="Combined transcript.*limit"):
        single_track.combine_dote_chunks(paths, output, [1] * 3)
    assert output.read_bytes() == expected_bytes


def test_single_chunk_budget_preserves_original_bytes(tmp_path, monkeypatch):
    source = chunk(tmp_path / "chunk.json")
    source.write_text(json.dumps(json.loads(source.read_text()), indent=2))
    output = tmp_path / "combined.json"
    monkeypatch.setattr(single_track, "COMBINED_MAX_LINES", 1, raising=False)
    monkeypatch.setattr(
        single_track, "COMBINED_MAX_BYTES", source.stat().st_size, raising=False
    )
    single_track.combine_dote_chunks([source], output, [0])
    assert output.read_bytes() == source.read_bytes()
    monkeypatch.setattr(single_track, "COMBINED_MAX_BYTES", source.stat().st_size - 1)
    with pytest.raises(ValueError, match="Combined transcript.*limit"):
        single_track.combine_dote_chunks([source], output, [0])
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("limit", ["lines", "bytes"])
def test_refused_replacement_keeps_all_completed_exports(episode, monkeypatch, limit):
    store, previous = episode
    monkeypatch.setattr(
        single_track,
        "COMBINED_MAX_LINES",
        1 if limit == "lines" else 100,
        raising=False,
    )
    monkeypatch.setattr(
        single_track,
        "COMBINED_MAX_BYTES",
        150 if limit == "bytes" else 10000,
        raising=False,
    )
    with pytest.raises(ValueError, match="Combined transcript.*limit"):
        single_track.transcribe(store.url, RecordingBackend("new", "new transcript"))
    assert all(
        (store.podcast_dir / p).read_bytes() == data for p, data in previous.items()
    )
