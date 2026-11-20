"""Cache components reserve bytes for exports and locks."""

import hashlib
import shutil
from pathlib import Path
from urllib.parse import quote

import pytest

from podcast_transcript import audio, single_track
from single_track_test import RecordingBackend


@pytest.mark.parametrize(
    "stem",
    [
        "a" * 248,
        "ä" * 124,
        "語" * 82,
        "a" * 200 + "_16khz",
        # Shortening drops the 4-byte characters, so the cut source name would
        # equal the resampled name unless the reserved check sees the cut name.
        "a" * 200 + "_16khz" + "\N{GRINNING FACE}" * 3,
    ],
)
@pytest.mark.parametrize("remote", [False, True])
def test_long_source_completes_with_byte_bounded_distinct_paths(
    tmp_path, monkeypatch, stem, remote
):
    source = tmp_path / f"{stem}.mp3"
    source.write_bytes(b"synthetic audio")
    url = f"https://example.test/{quote(source.name)}" if remote else str(source)
    monkeypatch.setattr(single_track.settings, "transcript_dir", tmp_path / "cache")
    monkeypatch.setattr(
        audio, "download", lambda _, path: shutil.copyfile(source, path)
    )
    monkeypatch.setattr(audio, "resample_audio", shutil.copyfile)
    monkeypatch.setattr(audio, "get_audio_duration", lambda _: 1.0)
    store = audio.Audio(base_dir=single_track.settings.transcript_dir, url=url)
    paths = [
        store.episode_path,
        store.resampled_episode_path,
        *[store.output_path(s) for s in audio.TRANSCRIPT_OUTPUT_SUFFIXES],
    ]
    for path in paths:
        assert len(path.name.encode("utf-8")) <= 213
    assert len(set(paths)) == len(paths)
    assert len(f"{store.cache_key}.lock".encode("utf-8")) <= 255
    result = single_track.transcribe(url, RecordingBackend("first", "one"))
    single_track.transcribe(url, RecordingBackend("second", "two"))
    assert result["plain text"].read_text() == "two"
    assert store.episode_path.read_bytes() == b"synthetic audio"
    assert single_track.restore_last_success(url)["plain text"].read_text() == "one"


def test_long_percent_decoded_url_names_remain_distinct_and_stable(tmp_path):
    first = "https://example.test/" + quote("ä" * 300 + "A.mp3")
    second = "https://example.test/" + quote("ä" * 300 + "B.mp3")
    stores = [audio.Audio(base_dir=tmp_path, url=url) for url in [first, second]]
    assert stores[0].cache_key != stores[1].cache_key
    for store in stores:
        assert len(store.cache_key.encode("utf-8")) <= 250
        assert len(store.episode_path.name.encode("utf-8")) <= 213
        assert store.cache_key == audio.get_cache_key(store.url)


@pytest.mark.parametrize(
    "source",
    [
        "https://example.test/episode.final.mp3?token=synthetic",
        "/synthetic/episode.final.mp3",
    ],
)
def test_short_existing_cache_keys_and_paths_stay_compatible(tmp_path, source):
    stem = audio.get_title_from_string(source)
    digest = hashlib.sha256(audio.get_cache_source(source).encode("utf-8")).hexdigest()[
        :12
    ]
    store = audio.Audio(base_dir=tmp_path, url=source)
    assert store.cache_key == f"{stem}-{digest}"
    assert store.output_path(".txt").name == f"{stem}.txt"
    assert store.episode_path.name == Path(source.split("?")[0]).name
