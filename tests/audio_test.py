import pytest

from podcast_transcript.audio import is_url, get_title_from_string


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
