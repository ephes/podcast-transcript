import httpx
import pytest

from podcast_transcript.config import settings
from podcast_transcript import audio as audio_module
from podcast_transcript.audio import download, is_url, get_title_from_string


@pytest.mark.parametrize(
    "url, is_http_url",
    [
        ("https://example.com/test.mp3", True),
        ("foo/test.mp3", False),
    ],
)
def test_is_url(url, is_http_url):
    # Given an url from parameters

    # When the url is checked
    result = is_url(url)

    # Then the result is as expected
    assert result == is_http_url


@pytest.mark.parametrize(
    "url, title",
    [
        ("https://example.com/test.mp3", "test"),
        ("foo/blub.mp3", "blub"),
    ],
)
def test_get_title_from_string(url, title):
    # Given a url from parameters, when the title is extracted
    from_string = get_title_from_string(url)

    # Then the title is as expected
    assert from_string == title


@pytest.fixture
def mock_http(mocker):
    """Route the download client through an httpx.MockTransport."""
    routes: dict[str, httpx.Response] = {}
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return routes[str(request.url)]

    real_client = httpx.Client

    def make_client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    mocker.patch.object(audio_module.httpx, "Client", side_effect=make_client)
    return routes, seen


def test_download_follows_redirects(mock_http, audio):
    routes, seen = mock_http
    routes["https://example.com/test.mp3"] = httpx.Response(
        302, headers={"Location": "https://cdn.example.com/real.mp3"}
    )
    routes["https://cdn.example.com/real.mp3"] = httpx.Response(
        200, content=b"audio content"
    )

    download(audio.url, audio.episode_path)

    assert seen == ["https://example.com/test.mp3", "https://cdn.example.com/real.mp3"]
    assert audio.episode_path.read_bytes() == b"audio content"
    assert [p.name for p in audio.episode_path.parent.iterdir()] == [
        audio.episode_path.name
    ]


def test_download_http_error_raises_and_caches_nothing(mock_http, audio):
    routes, _ = mock_http
    routes["https://example.com/test.mp3"] = httpx.Response(404, content=b"not found")

    with pytest.raises(httpx.HTTPStatusError):
        audio.make_sure_audio_file_exists()

    assert not audio.episode_path.exists()
    assert list(audio.episode_chunks_dir.iterdir()) == []


def test_download_interrupted_stream_caches_nothing(mocker, audio):
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"first half"
            raise httpx.ReadError("connection reset")

    real_client = httpx.Client
    mocker.patch.object(
        audio_module.httpx,
        "Client",
        side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=BrokenStream())
            ),
            **kwargs,
        ),
    )

    with pytest.raises(httpx.ReadError):
        audio.make_sure_audio_file_exists()

    assert not audio.episode_path.exists()
    assert list(audio.episode_chunks_dir.iterdir()) == []


@pytest.mark.parametrize(
    "url, filename",
    [
        ("https://example.test/audio.mp3?token=x/y#secret/z", "audio.mp3"),
        ("https://example.test?token=secret", "episode.mp3"),
        ("https://example.test/", "episode.mp3"),
        ("https://example.test/my%20audio.mp3", "my_audio.mp3"),
        ("https://example.test/%2F..%5Csecret.mp3", "_.._secret.mp3"),
        ("https://example.test/%2E%2E", "episode.mp3"),
        ("https://example.test/chunk_000.mp3?token=x/y", "episode_chunk_000.mp3"),
    ],
)
def test_url_download_filename_uses_safe_path_only(tmp_path, mocker, url, filename):
    from podcast_transcript.audio import Audio

    audio = Audio(base_dir=tmp_path, url=url)
    download_mock = mocker.patch("podcast_transcript.audio.download")
    audio.make_sure_audio_file_exists()
    download_mock.assert_called_once_with(url, audio.episode_chunks_dir / filename)
    assert audio.episode_path.parent == audio.episode_chunks_dir
    assert "?" not in audio.episode_path.name
    assert "#" not in audio.episode_path.name
    assert audio.podcast_dir.parent == tmp_path


@pytest.mark.parametrize("relative", [True, False])
def test_short_audio_cache_root_with_spaces(tmp_path, monkeypatch, mocker, relative):
    import shutil
    from pathlib import Path
    from podcast_transcript.audio import Audio

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.mp3"
    source.write_bytes(b"synthetic audio")
    root = Path("cache with spaces") if relative else tmp_path / "cache with spaces"
    audio = Audio(base_dir=root, url=str(source))
    mocker.patch("podcast_transcript.audio.resample_audio", shutil.copyfile)
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    chunks = audio.prepare_audio_for_transcription()
    assert len(chunks) == 1
    assert chunks[0].is_symlink()
    assert chunks[0].read_bytes() == b"synthetic audio"
    assert chunks[0].resolve() == audio.resampled_episode_path.resolve()
    assert audio.prepare_audio_for_transcription() == chunks


@pytest.mark.parametrize(
    "output",
    [
        "not json",
        "[]",
        "null",
        "{}",
        '{"format": []}',
        '{"format": {}}',
        '{"format": {"duration": null}}',
        '{"format": {"duration": true}}',
        '{"format": {"duration": "NaN"}}',
        '{"format": {"duration": "inf"}}',
        '{"format": {"duration": "-1"}}',
    ],
)
def test_ffprobe_rejects_invalid_duration(mocker, output):
    import subprocess

    mocker.patch(
        "subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=output)
    )
    with pytest.raises(RuntimeError, match="ffprobe"):
        audio_module.get_audio_duration("synthetic.mp3")


@pytest.mark.parametrize("duration", ["0", "12.5"])
def test_ffprobe_accepts_finite_nonnegative_duration(mocker, duration):
    import subprocess
    import json

    run = mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            [], 0, stdout=json.dumps({"format": {"duration": duration}})
        ),
    )
    assert audio_module.get_audio_duration("synthetic.mp3") == float(duration)
    assert run.call_args.kwargs["errors"] == "replace"
    assert run.call_args.kwargs["timeout"] == 60.0
    assert run.call_args.kwargs["stdin"] == subprocess.DEVNULL


@pytest.mark.parametrize("timeout", [True, False])
def test_ffprobe_process_errors_are_controlled_and_bounded(mocker, timeout):
    import subprocess

    error = (
        subprocess.TimeoutExpired(["ffprobe"], 60, stderr=b"x" * 10000)
        if timeout
        else subprocess.CalledProcessError(1, ["ffprobe"], stderr="x" * 10000)
    )
    mocker.patch("subprocess.run", side_effect=error)
    with pytest.raises(RuntimeError, match="ffprobe") as exc:
        audio_module.get_audio_duration("synthetic.mp3")
    assert len(str(exc.value)) < 2300


@pytest.mark.parametrize("length", [None, "20", "0"])
def test_source_download_byte_budget_preserves_target(mocker, audio, length):
    mocker.patch.object(settings, "transcript_source_max_bytes", "6", create=True)

    class Stream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield b"1234"
            yield b"5678"

        def close(self):
            self.closed = True

    stream = Stream()
    real_client = httpx.Client
    response = httpx.Response(
        200, headers={} if length is None else {"Content-Length": length}, stream=stream
    )
    mocker.patch.object(
        audio_module.httpx,
        "Client",
        side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(lambda request: response), **kwargs
        ),
    )
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"old audio")
    with pytest.raises(RuntimeError, match="TRANSCRIPT_SOURCE_MAX_BYTES"):
        download(audio.url, audio.episode_path)
    assert stream.closed
    assert audio.episode_path.read_bytes() == b"old audio"
    assert list(audio.episode_chunks_dir.iterdir()) == [audio.episode_path]


@pytest.mark.parametrize("size", [5, 6])
def test_source_download_near_byte_limit_succeeds(mocker, mock_http, audio, size):
    mocker.patch.object(settings, "transcript_source_max_bytes", "6", create=True)
    routes, _ = mock_http
    routes[audio.url] = httpx.Response(200, content=b"x" * size)
    download(audio.url, audio.episode_path)
    assert audio.episode_path.read_bytes() == b"x" * size


@pytest.mark.parametrize("stage", ["headers", "drip", "eof"])
def test_source_download_total_budget_closes_and_cleans_up(mocker, audio, stage):
    mocker.patch.object(settings, "transcript_source_total_timeout", "1", create=True)
    clock = [0.0]
    mocker.patch.object(
        audio_module, "monotonic", side_effect=lambda: clock[0], create=True
    )

    class Stream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            clock[0] += 0.6
            yield b"first"
            clock[0] += 0.6
            if stage != "eof":
                yield b"second"

        def close(self):
            self.closed = True

    stream = Stream()

    def handler(request):
        if stage == "headers":
            clock[0] = 1.1
        return httpx.Response(200, stream=stream)

    real_client = httpx.Client
    mocker.patch.object(
        audio_module.httpx,
        "Client",
        side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"old audio")
    with pytest.raises(RuntimeError, match="TRANSCRIPT_SOURCE_TOTAL_TIMEOUT"):
        download(audio.url, audio.episode_path)
    assert stream.closed
    assert audio.episode_path.read_bytes() == b"old audio"
    assert list(audio.episode_chunks_dir.iterdir()) == [audio.episode_path]


def test_non_utf8_chunk_manifest_is_rebuilt(mocker, audio):
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.resampled_episode_path.write_bytes(b"synthetic")
    audio.chunks_manifest_path.write_bytes(b"\xff\xfe")
    chunks = audio.split_into_chunks()
    assert chunks[0].read_bytes() == b"synthetic"
    assert audio._load_chunk_manifest() == chunks
