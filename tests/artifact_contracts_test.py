"""Transcript admission and cached timecodes, using synthetic artifacts only."""

import json
from types import SimpleNamespace

import httpx
import pytest

from podcast_transcript import backends, single_track as track


INVALID_ENVELOPES = [
    [],
    None,
    {},
    {"text": "only"},
    {"segments": None},
    {"segments": {}},
    {"segments": [None]},
    {"segments": [{"start": 0, "end": 1}]},
    {"segments": [{"start": 0, "end": 1, "text": 42}]},
    {"segments": [], "text": []},
]


def backend(name):
    if name == "groq":
        return backends.Groq(api_key="fake", model_name=None, language="en", prompt="p")
    return backends.Voxhelm(
        api_base="https://voxhelm.test",
        api_key="fake",
        model_name=None,
        language=None,
        prompt=None,
    )


@pytest.mark.parametrize("name", ["groq", "voxhelm"])
@pytest.mark.parametrize("invalid", INVALID_ENVELOPES)
def test_bad_success_never_published_then_retry_recovers(
    monkeypatch, tmp_path, name, invalid
):
    results = iter([invalid, {"segments": []}])
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: real_client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, content=json.dumps(next(results)).encode()
                )
            ),
            **kw,
        ),
    )
    chunk = tmp_path / "chunk.mp3"
    chunk.write_bytes(b"audio")
    out = chunk.with_suffix(".json")
    with pytest.raises(ValueError, match="Invalid transcript"):
        backend(name).transcribe(chunk, out)
    assert not out.exists()
    backend(name).transcribe(chunk, out)
    assert json.loads(out.read_text()) == {"segments": []}


@pytest.mark.parametrize("invalid", INVALID_ENVELOPES)
def test_invalid_raw_cache_is_rejected(monkeypatch, tmp_path, invalid):
    raw = tmp_path / "chunk.json"
    raw.write_text(json.dumps(invalid))

    def unexpected(*args):
        pytest.fail("cached data must be validated before proceeding")

    with pytest.raises(ValueError, match=r"chunk\.json.*Invalid transcript"):
        track.audio_chunks_to_text(
            SimpleNamespace(transcribe=unexpected), [raw.with_suffix(".mp3")]
        )
    assert not raw.with_suffix(".dote.json").exists()


def test_mlx_invalid_result_preserves_previous_transcript(fake_mlx, tmp_path):
    fake_mlx(
        """
        def transcribe(*args, **kwargs):
            return {"text": "only"}
        """
    )
    out = tmp_path / "chunk.json"
    out.write_text("old")
    with pytest.raises(ValueError, match="Invalid transcript"):
        backends.MLX(model_name=None).transcribe(tmp_path / "chunk.mp3", out)
    assert out.read_text() == "old"


@pytest.mark.parametrize(
    "start,end",
    [
        (True, 1),
        (0, False),
        ("0", 1),
        (0, "1"),
        (None, 1),
        (float("nan"), 1),
        (0, float("inf")),
        (-1, 1),
        (0, -1),
        (2, 1),
        (1e308, 1e308),
        (10**400, 10**400),
    ],
)
def test_segment_time_domain_prevents_dote_publication(tmp_path, start, end):
    raw = tmp_path / "chunk.json"
    raw.write_text(
        json.dumps({"segments": [{"start": start, "end": end, "text": "private"}]})
    )
    with pytest.raises(ValueError, match="segment 1") as error:
        track.whisper_text_chunks_to_dote([raw])
    assert len(str(error.value)) < 200
    assert "private" not in str(error.value)
    assert not raw.with_suffix(".dote.json").exists()
    with pytest.raises(ValueError, match="segment 1"):
        track.whisper_to_dote([{"start": start, "end": end, "text": "private"}])


def test_fractional_and_zero_length_segments(tmp_path):
    raw = tmp_path / "chunk.json"
    raw.write_text(
        json.dumps(
            {"segments": [{"start": 90000.0006, "end": 90000.0006, "text": " x "}]}
        )
    )
    [out] = track.whisper_text_chunks_to_dote([raw])
    line = json.loads(out.read_text())["lines"][0]
    assert line["startTime"] == line["endTime"] == "25:00:00,001"
    assert line["text"] == "x"


def dote(path, start, end):
    path.write_text(
        json.dumps(
            {
                "lines": [
                    {
                        "startTime": start,
                        "endTime": end,
                        "text": "hello",
                        "speakerDesignation": "",
                    }
                ]
            }
        )
    )
    return path


def combine_one(source, out):
    track.combine_dote_chunks([source], out, [0])


def combine_multiple(source, out):
    track.combine_dote_chunks([source, source], out, [1, 1])


EXPORTERS = [
    combine_one,
    combine_multiple,
    track.convert_dote_to_podlove,
    track.convert_to_webvtt,
    track.convert_to_plaintext,
]


@pytest.mark.parametrize("exporter", EXPORTERS)
@pytest.mark.parametrize(
    "start,end",
    [
        ("00:99:00,000junk", "01:00:00,000"),
        ("00:00:60,000", "01:00:00,000"),
        ("00:60:00,000", "01:00:00,000"),
        ("00:00:00,1000", "01:00:00,000"),
        ("0:00:00,000", "01:00:00,000"),
        ("00:0:00,000", "01:00:00,000"),
        ("00:00:00,00", "01:00:00,000"),
        ("00:00:00,000\n", "01:00:00,000"),
        ("00:00:02,000", "00:00:01,000"),
        (None, "00:00:01,000"),
        ("00:00:00,000", "junk"),
    ],
)
def test_cached_timecode_rejected_before_each_publication(
    tmp_path, exporter, start, end
):
    source = dote(tmp_path / "chunk.dote.json", start, end)
    out = tmp_path / "out"
    out.write_bytes(b"previous")
    with pytest.raises(ValueError, match=r"chunk\.dote\.json.*DOTe"):
        exporter(source, out)
    assert out.read_bytes() == b"previous"


@pytest.mark.parametrize("exporter", EXPORTERS)
def test_multiday_timecodes_supported_by_all_exports(tmp_path, exporter):
    source = dote(tmp_path / "chunk.dote.json", "125:00:00,001", "125:00:00,001")
    out = tmp_path / "out"
    exporter(source, out)
    assert out.exists()
    assert "hello" in out.read_text()


def test_whisper_cpp_invalid_times_preserve_raw_output(tmp_path):
    source = tmp_path / "cpp.json"
    source.write_text(
        json.dumps(
            {
                "transcription": [
                    {
                        "offsets": {"from": 0},
                        "timestamps": {"from": "00:00:02,000", "to": "00:00:01,000"},
                        "text": "private",
                    }
                ]
            }
        )
    )
    out = tmp_path / "raw.json"
    out.write_bytes(b"previous")
    with pytest.raises(ValueError, match="segment 1"):
        backends.WhisperCpp().convert_output_format(source, out)
    assert out.read_bytes() == b"previous"


@pytest.mark.parametrize("exporter", EXPORTERS)
def test_dote_speaker_is_optional_but_must_be_string(tmp_path, exporter):
    source = dote(tmp_path / "chunk.dote.json", "00:00:00,000", "00:00:01,000")
    data = json.loads(source.read_text())
    del data["lines"][0]["speakerDesignation"]
    source.write_text(json.dumps(data))
    out = tmp_path / "out"
    exporter(source, out)
    original = out.read_bytes()
    if exporter is track.convert_dote_to_podlove:
        assert json.loads(original)["transcripts"][0]["speaker"] == ""
    data["lines"][0]["speakerDesignation"] = 42
    source.write_text(json.dumps(data))
    with pytest.raises(
        ValueError, match=r"chunk\.dote\.json.*line 1.*speakerDesignation"
    ):
        exporter(source, out)
    assert out.read_bytes() == original


def test_malformed_cache_errors_name_files(tmp_path):
    for name, load in [
        ("chunk_007.json", track.load_whisper_transcript),
        ("chunk_007.dote.json", track.load_dote),
    ]:
        path = tmp_path / name
        path.write_text("{")
        with pytest.raises(ValueError, match=name):
            load(path)
