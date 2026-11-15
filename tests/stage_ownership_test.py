"""Only admitted stages owned by this episode may be reclaimed."""

import json
from pathlib import Path

import pytest

from podcast_transcript import single_track


def stage_manifest(path, audio):
    info = path.stat()
    return {
        "version": 1,
        "episode": audio.cache_key,
        "name": path.name,
        "device": info.st_dev,
        "inode": info.st_ino,
    }


def make_stage(audio, name=".replacement-killed"):
    path = audio.podcast_dir / name
    path.mkdir(parents=True)
    (path / ".stage-owner.json").write_text(json.dumps(stage_manifest(path, audio)))
    (path / "sentinel").write_text("synthetic")
    return path


@pytest.mark.parametrize(
    "damage",
    ["absent", "invalid", "oversized", "wrong-episode", "copied", "symlink-marker"],
)
def test_unknown_matching_directory_survives(audio, damage):
    path = make_stage(audio)
    marker = path / ".stage-owner.json"
    if damage == "absent":
        marker.unlink()
    elif damage == "invalid":
        marker.write_text('{"version":1,"version":1}')
    elif damage == "oversized":
        marker.write_bytes(b" " * 65537)
    elif damage == "wrong-episode":
        data = stage_manifest(path, audio)
        data["episode"] = "other"
        marker.write_text(json.dumps(data))
    elif damage == "copied":
        data = stage_manifest(path, audio)
        data["inode"] += 1
        marker.write_text(json.dumps(data))
    else:
        other = audio.podcast_dir / "other.json"
        marker.rename(other)
        marker.symlink_to(other)
    single_track._remove_abandoned_staging(audio)
    assert (path / "sentinel").read_text() == "synthetic"


def test_owned_stage_and_symlink_contents_are_reclaimed(audio, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("synthetic")
    for name in (".replacement-killed", ".backup-killed"):
        stage = make_stage(audio, name)
        (stage / "link").symlink_to(outside, target_is_directory=True)
    alias = audio.podcast_dir / ".replacement-symlink"
    alias.symlink_to(outside, target_is_directory=True)
    single_track._remove_abandoned_staging(audio)
    assert not (audio.podcast_dir / ".replacement-killed").exists()
    assert not (audio.podcast_dir / ".backup-killed").exists()
    assert alias.is_symlink()
    assert (outside / "keep").read_text() == "synthetic"


def test_exchanged_stage_cannot_redirect_cleanup(audio, monkeypatch):
    path = make_stage(audio)
    moved = audio.podcast_dir / "moved-owned-stage"
    read = single_track.load_control_json

    # Exchange the named entry after ownership was read. Cleanup must stay bound
    # to the opened stage or refuse it, never traverse the replacement directory.
    def exchange(marker, *args, **kwargs):
        result = read(marker, *args, **kwargs)
        if Path(marker).name == ".stage-owner.json" and not moved.exists():
            path.rename(moved)
            path.mkdir()
            (path / "unrelated").write_text("keep")
        return result

    monkeypatch.setattr(single_track, "load_control_json", exchange)
    single_track._remove_abandoned_staging(audio)
    assert (path / "unrelated").read_text() == "keep"


@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_exchange_after_admission_does_not_redirect_descriptor_cleanup(
    audio, monkeypatch, replacement
):
    path = make_stage(audio)
    inode = path.stat().st_ino
    moved = audio.podcast_dir / "opened-owned-stage"
    outside = audio.podcast_dir / "unrelated"
    outside.mkdir()
    (outside / "keep").write_text("synthetic")
    scandir = single_track.os.scandir
    exchanged = False

    def exchange(directory):
        nonlocal exchanged
        if (
            isinstance(directory, int)
            and single_track.os.fstat(directory).st_ino == inode
            and not exchanged
        ):
            exchanged = True
            path.rename(moved)
            if replacement == "symlink":
                path.symlink_to(outside, target_is_directory=True)
            else:
                path.mkdir()
                (path / "keep").write_text("synthetic")
        return scandir(directory)

    monkeypatch.setattr(single_track.os, "scandir", exchange)
    single_track._remove_abandoned_staging(audio)
    assert exchanged
    assert (path / "keep").read_text() == "synthetic"
    assert (outside / "keep").read_text() == "synthetic"


def test_new_pipeline_stages_have_ownership_manifests(episode, monkeypatch):
    from single_track_test import RecordingBackend

    store, _ = episode
    copy = single_track.copy_file_atomic
    seen = set()

    def verify(source, target):
        for path in (source, target):
            for parent in path.parents:
                if parent.name.startswith(single_track.STAGING_PREFIXES):
                    marker = json.loads((parent / ".stage-owner.json").read_text())
                    assert marker == stage_manifest(parent, store)
                    seen.add(parent.name.split("-", 1)[0])
                    break
        copy(source, target)

    monkeypatch.setattr(single_track, "copy_file_atomic", verify)
    single_track.transcribe(store.url, RecordingBackend("new", "new"))
    assert seen == {".replacement", ".backup"}
