import json
from pathlib import Path

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

    with pytest.raises(RuntimeError, match="Source download failed.*status 404"):
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

    with pytest.raises(RuntimeError, match="Source download failed.*ReadError"):
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
    mocker.patch(
        "podcast_transcript.audio.capture_checked", return_value=output.encode()
    )
    with pytest.raises(RuntimeError, match="ffprobe"):
        audio_module.get_audio_duration("synthetic.mp3")


@pytest.mark.parametrize("duration", ["0", "12.5"])
def test_ffprobe_accepts_finite_nonnegative_duration(mocker, duration):
    import json

    run = mocker.patch(
        "podcast_transcript.audio.capture_checked",
        return_value=json.dumps({"format": {"duration": duration}}).encode(),
    )
    assert audio_module.get_audio_duration("synthetic.mp3") == float(duration)
    assert run.call_args.kwargs["timeout"] == 60.0


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


def test_non_utf8_chunk_manifest_preserves_generation(mocker, audio):
    mocker.patch("podcast_transcript.audio.get_audio_duration", return_value=1.0)
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.resampled_episode_path.write_bytes(b"synthetic")
    audio.chunks_manifest_path.write_bytes(b"\xff\xfe")
    with pytest.raises(RuntimeError, match="control JSON"):
        audio.split_into_chunks()
    assert audio.resampled_episode_path.read_bytes() == b"synthetic"
    assert audio.chunks_manifest_path.read_bytes() == b"\xff\xfe"


@pytest.mark.parametrize(
    "entrypoint", ["prepare_audio_for_transcription", "split_into_chunks"]
)
def test_oversized_chunk_inventory_refused_without_chunk_stats(
    audio, mocker, monkeypatch, entrypoint
):
    monkeypatch.setattr(audio_module, "MAX_CHUNK_COUNT", 2, raising=False)
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"source")
    audio.resampled_episode_path.write_bytes(b"resampled")
    output = audio.output_path(".txt")
    output.write_bytes(b"previous transcript")
    names = ["chunk_000.mp3", "chunk_001.mp3", "chunk_002.mp3"]
    paths = [audio.episode_chunks_dir / name for name in names]
    for path in reversed(paths):
        path.write_bytes(b"chunk")
    mocker.patch.object(audio_module, "get_audio_duration", return_value=1.0)
    make_audio = mocker.spy(audio, "make_sure_audio_file_exists")
    manifest = json.dumps(names)
    audio.chunks_manifest_path.write_text(manifest)
    probes = [mocker.spy(Path, method) for method in ("is_file", "stat", "is_symlink")]

    def chunk_probes():
        return {
            call.args[0]
            for probe in probes
            for call in probe.call_args_list
            if call.args[0] in paths
        }

    with pytest.raises(RuntimeError, match=r"chunks/chunks\.json.*2 chunks"):
        getattr(audio, entrypoint)()

    make_audio.assert_not_called()
    assert not chunk_probes()
    assert output.read_bytes() == b"previous transcript"
    assert audio.chunks_manifest_path.read_text() == manifest

    audio.chunks_manifest_path.write_text(json.dumps(names[:2]))
    assert getattr(audio, entrypoint)() == paths[:2]
    assert chunk_probes() == set(paths[:2])


def test_generated_chunk_inventory_refused_before_validation(
    audio, mocker, monkeypatch
):
    monkeypatch.setattr(audio_module, "MAX_CHUNK_COUNT", 2, raising=False)
    audio.episode_chunks_dir.mkdir(parents=True)
    resampled = audio.resampled_episode_path
    resampled.write_bytes(b"resampled")
    probed = []

    def probe(path):
        probed.append(path)
        return 7201.0 if path == resampled else 1.0

    def split(command, **kwargs):
        for index in range(3):
            Path(command[-1] % index).write_bytes(b"chunk")

    mocker.patch.object(audio_module, "get_audio_duration", side_effect=probe)
    mocker.patch.object(audio_module, "run_checked", side_effect=split)
    with pytest.raises(RuntimeError, match="more than 2 chunks"):
        audio.split_into_chunks()

    assert probed == [resampled]
    assert list(audio.episode_chunks_dir.iterdir()) == [resampled]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "status, headers, body",
    [
        (206, {"Content-Range": "bytes 0-2/100", "Content-Length": "3"}, b"abc"),
        (206, {}, b"abc"),
        (200, {"Content-Range": "bytes 0-2/100"}, b"abc"),
        (200, {"Content-Range": ""}, b"abc"),
        (204, {}, b""),
        (205, {}, b""),
        (200, {}, b""),
        (200, {"Content-Length": "0"}, b""),
    ],
)
def test_source_download_refuses_incomplete_audio(
    mock_http, audio, existing, status, headers, body
):
    class Stream(httpx.SyncByteStream):
        read = False
        closed = False

        def __iter__(self):
            self.read = True
            yield body

        def close(self):
            self.closed = True

    stream = Stream()
    routes, _ = mock_http
    routes[audio.url] = httpx.Response(status, headers=headers, stream=stream)
    audio.episode_chunks_dir.mkdir(parents=True)
    if existing:
        audio.episode_path.write_bytes(b"previous complete episode")

    with pytest.raises(RuntimeError, match="Source download requires"):
        download(audio.url, audio.episode_path)

    assert stream.closed
    if status != 200 or "Content-Range" in headers:
        assert not stream.read
    if existing:
        assert audio.episode_path.read_bytes() == b"previous complete episode"
    else:
        assert not audio.episode_path.exists()
    assert list(audio.episode_chunks_dir.iterdir()) == (
        [audio.episode_path] if existing else []
    )


@pytest.mark.parametrize("status, body", [(206, b"abc"), (204, b""), (200, b"")])
def test_refused_source_never_reaches_preparation(
    mock_http, mocker, audio, status, body
):
    routes, _ = mock_http
    routes[audio.url] = httpx.Response(status, content=body)
    resample = mocker.patch.object(audio, "make_sure_audio_file_is_resampled")
    split = mocker.patch.object(audio, "split_into_chunks")

    with pytest.raises(RuntimeError, match="Source download requires"):
        audio.prepare_audio_for_transcription()

    resample.assert_not_called()
    split.assert_not_called()
    assert list(audio.episode_chunks_dir.iterdir()) == []


@pytest.mark.parametrize(
    "lengths, refusal",
    [
        (["5"], "does not match Content-Length"),
        (["2"], "does not match Content-Length"),
        (["3", "3"], "one decimal Content-Length"),
        (["3, 3"], "one decimal Content-Length"),
        (["+3"], "one decimal Content-Length"),
        ([""], "one decimal Content-Length"),
        (["003"], None),
        ([], None),
    ],
)
def test_source_download_requires_length_agreement(mock_http, audio, lengths, refusal):
    class Stream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield b"abc"

        def close(self):
            self.closed = True

    stream = Stream()
    routes, _ = mock_http
    routes[audio.url] = httpx.Response(
        200, headers=[("Content-Length", length) for length in lengths], stream=stream
    )
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"previous")
    if refusal:
        with pytest.raises(RuntimeError, match=refusal):
            download(audio.url, audio.episode_path)
        assert audio.episode_path.read_bytes() == b"previous"
    else:
        download(audio.url, audio.episode_path)
        assert audio.episode_path.read_bytes() == b"abc"
    assert stream.closed
    assert list(audio.episode_chunks_dir.iterdir()) == [audio.episode_path]


@pytest.mark.parametrize("outcome", ["success", "limit", "timeout"])
def test_source_download_closes_redirects_without_reading(mocker, audio, outcome):
    class RedirectBody(httpx.SyncByteStream):
        reads = 0
        closes = 0

        def __iter__(self):
            self.reads += 1
            yield b"x" * 256

        def close(self):
            self.closes += 1

    clock = [0.0]
    mocker.patch.object(audio_module, "monotonic", side_effect=lambda: clock[0])
    mocker.patch.object(settings, "transcript_source_total_timeout", "5")
    mocker.patch.object(settings, "transcript_source_max_bytes", "8")
    requests = []
    streams = []

    def handler(request):
        assert all(stream.closes == 1 for stream in streams)
        requests.append(request)
        if len(requests) == 3 and outcome == "success":
            return httpx.Response(200, content=b"audio")
        if len(requests) == 2 and outcome == "timeout":
            clock[0] = 5.0
        location = {
            1: "/relative.mp3",
            2: "https://cdn.example.com/final.mp3",
            3: "/one-hop-too-many.mp3",
        }[len(requests)]
        stream = RedirectBody()
        streams.append(stream)
        return httpx.Response(
            302,
            headers={"Location": location, "Content-Encoding": "gzip"},
            stream=stream,
        )

    real_client = httpx.Client
    mocker.patch.object(
        audio_module.httpx,
        "Client",
        side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(handler), max_redirects=2, **kwargs
        ),
    )
    audio.episode_chunks_dir.mkdir(parents=True)
    audio.episode_path.write_bytes(b"previous")
    url = "https://synthetic:secret@example.com/test.mp3"
    if outcome == "success":
        download(url, audio.episode_path)
        assert audio.episode_path.read_bytes() == b"audio"
    else:
        error = (
            "TooManyRedirects"
            if outcome == "limit"
            else "TRANSCRIPT_SOURCE_TOTAL_TIMEOUT"
        )
        with pytest.raises(RuntimeError, match=error):
            download(url, audio.episode_path)
        assert audio.episode_path.read_bytes() == b"previous"

    assert all(stream.reads == 0 and stream.closes == 1 for stream in streams)
    expected_urls = [
        ("example.com", "/test.mp3"),
        ("example.com", "/relative.mp3"),
        ("cdn.example.com", "/final.mp3"),
    ]
    assert [(r.url.host, r.url.path) for r in requests] == (
        expected_urls[:2] if outcome == "timeout" else expected_urls
    )
    assert requests[0].headers["Authorization"] == requests[1].headers["Authorization"]
    if len(requests) == 3:
        assert "Authorization" not in requests[2].headers
    for request in requests:
        assert set(request.extensions["timeout"].values()) == {5.0}
        assert request.headers["Accept-Encoding"] == "identity"
    assert list(audio.episode_chunks_dir.iterdir()) == [audio.episode_path]
