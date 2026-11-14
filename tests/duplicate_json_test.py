"""Ambiguous cache JSON is refused without losing completed evidence."""

import pytest

from podcast_transcript.transcript_validation import load_dote, load_whisper_transcript


@pytest.mark.parametrize(
    "members",
    [
        '"text":"private-a","text":"private-b","start":0,"end":1',
        '"text":"private-b","text":"private-a","start":0,"end":1',
        '"text":"ok","start":0,"start":0.5,"end":1',
    ],
)
def test_raw_duplicate_members(tmp_path, members):
    path = tmp_path / "raw.json"
    path.write_text('{"segments":[{' + members + "}]}")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid transcript JSON") as error:
        load_whisper_transcript(path)
    assert "private-" not in str(error.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "members",
    [
        '"segments":[],"segments":[]',
        '"extra":{"backend":"a","backend":"b"},"segments":[]',
    ],
)
def test_raw_top_level_and_extension_duplicates(tmp_path, members):
    path = tmp_path / "raw.json"
    path.write_text("{" + members + "}")
    with pytest.raises(ValueError, match="Invalid transcript JSON"):
        load_whisper_transcript(path)


@pytest.mark.parametrize(
    "first,second", [("00:00:00,000", "00:00:00,500"), ("00:00:00,500", "00:00:00,000")]
)
def test_dote_duplicate_members(tmp_path, first, second):
    path = tmp_path / "cached.dote.json"
    path.write_text(
        '{"lines":[{"text":"private-text","startTime":"'
        + first
        + '","startTime":"'
        + second
        + '","endTime":"00:00:01,000"}]}'
    )
    with pytest.raises(ValueError, match="Invalid DOTe JSON") as error:
        load_dote(path)
    assert "private-text" not in str(error.value)


@pytest.mark.parametrize("first,second", [("old", "new"), ("new", "old")])
def test_cache_control_duplicates_preserve_generation(audio, first, second):
    audio.podcast_dir.mkdir(parents=True)
    output = audio.output_path(".txt")
    output.write_bytes(b"previous")
    audio.cache_metadata_path.write_text(
        '{"backend":"' + first + '","backend":"' + second + '"}'
    )
    before = audio.cache_metadata_path.read_bytes()
    with pytest.raises(RuntimeError, match="control JSON"):
        audio.ensure_transcript_cache_matches({"backend": "new"})
    assert output.read_bytes() == b"previous"
    assert audio.cache_metadata_path.read_bytes() == before
