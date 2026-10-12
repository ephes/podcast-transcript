import httpx
import pytest

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
