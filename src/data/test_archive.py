# noqa: INP001
"""Tests for Archiver._archive_file() — freshness-aware archive copying."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from archive import Archiver


@pytest.fixture()
def arch():
    return Archiver(add_handlers=False, clobber=False)


def _touch_with_mtime(path: Path, content: str, mtime: float) -> None:
    path.write_text(content)
    os.utime(path, (mtime, mtime))


class TestArchiveFile:
    def test_copies_when_dst_missing(self, arch, tmp_path):
        src = tmp_path / "src.nc"
        dst = tmp_path / "dst.nc"
        src.write_text("fresh data")

        arch._archive_file(src, dst)

        assert dst.exists()  # noqa: S101
        assert dst.read_text() == "fresh data"  # noqa: S101

    def test_noop_when_src_missing(self, arch, tmp_path):
        src = tmp_path / "ghost.nc"
        dst = tmp_path / "dst.nc"

        arch._archive_file(src, dst)

        assert not dst.exists()  # noqa: S101

    def test_skips_when_dst_is_fresh(self, arch, tmp_path):
        """dst newer than src (the common case: dst was archived after src was
        last written) must be left alone — this is the normal "already
        archived, nothing changed" case."""
        src = tmp_path / "src.nc"
        dst = tmp_path / "dst.nc"
        _touch_with_mtime(src, "old data", mtime=1000)
        _touch_with_mtime(dst, "already archived", mtime=2000)

        arch._archive_file(src, dst)

        assert dst.read_text() == "already archived"  # noqa: S101

    def test_recopies_when_dst_is_stale(self, arch, tmp_path):
        """dst older than src (mission was reprocessed after it was archived)
        must be refreshed even without --clobber — this is the regression
        this test guards against: a stale archived copy silently never being
        updated."""
        src = tmp_path / "src.nc"
        dst = tmp_path / "dst.nc"
        _touch_with_mtime(src, "reprocessed data", mtime=2000)
        _touch_with_mtime(dst, "stale archived data", mtime=1000)

        arch._archive_file(src, dst)

        assert dst.read_text() == "reprocessed data"  # noqa: S101

    def test_clobber_overwrites_even_when_dst_is_fresh(self, tmp_path):
        """--clobber must force a copy regardless of freshness."""
        arch = Archiver(add_handlers=False, clobber=True)
        src = tmp_path / "src.nc"
        dst = tmp_path / "dst.nc"
        _touch_with_mtime(src, "new data", mtime=1000)
        _touch_with_mtime(dst, "old data but newer mtime", mtime=2000)

        arch._archive_file(src, dst)

        assert dst.read_text() == "new data"  # noqa: S101


class TestCopySbdToLRAUV:
    """copy_sbd_to_LRAUV() now delegates to _archive_file() for every file,
    so it must pick up freshness the same way copy_to_LRAUV() does."""

    def test_recopies_stale_product_without_clobber(self, arch, tmp_path):
        base_lrauv_path = tmp_path / "local"
        lrauv_vol = tmp_path / "vol"
        rel = Path("ahi/realtime/sbdlogs/2026/20260317_20260318")
        src_dir = base_lrauv_path / rel
        dst_dir = lrauv_vol / rel
        src_dir.mkdir(parents=True)
        dst_dir.mkdir(parents=True)

        stem = "ahi_20260317_20260318_sbd_1S"
        _touch_with_mtime(src_dir / f"{stem}.nc", "reprocessed", mtime=2000)
        _touch_with_mtime(dst_dir / f"{stem}.nc", "stale", mtime=1000)

        with (
            patch("archive.BASE_LRAUV_PATH", base_lrauv_path),
            patch("archive.LRAUV_VOL", str(lrauv_vol)),
        ):
            arch.copy_sbd_to_LRAUV(src_dir / f"{stem}.nc")

        assert (dst_dir / f"{stem}.nc").read_text() == "reprocessed"  # noqa: S101

    def test_skips_fresh_product_without_clobber(self, arch, tmp_path):
        base_lrauv_path = tmp_path / "local"
        lrauv_vol = tmp_path / "vol"
        rel = Path("ahi/realtime/sbdlogs/2026/20260317_20260318")
        src_dir = base_lrauv_path / rel
        dst_dir = lrauv_vol / rel
        src_dir.mkdir(parents=True)
        dst_dir.mkdir(parents=True)

        stem = "ahi_20260317_20260318_sbd_1S"
        _touch_with_mtime(src_dir / f"{stem}.nc", "old", mtime=1000)
        _touch_with_mtime(dst_dir / f"{stem}.nc", "already archived", mtime=2000)

        with (
            patch("archive.BASE_LRAUV_PATH", base_lrauv_path),
            patch("archive.LRAUV_VOL", str(lrauv_vol)),
        ):
            arch.copy_sbd_to_LRAUV(src_dir / f"{stem}.nc")

        assert (dst_dir / f"{stem}.nc").read_text() == "already archived"  # noqa: S101


class TestCopyLrauvDeployment:
    """copy_lrauv_deployment() now delegates to _archive_file() for every file."""

    def test_recopies_stale_index_without_clobber(self, arch, tmp_path):
        base_lrauv_path = tmp_path / "local"
        lrauv_vol = tmp_path / "vol"
        rel = Path("ahi/missionlogs/2026/20260317_20260318")
        deployment_dir = base_lrauv_path / rel
        dst_dir = lrauv_vol / rel
        deployment_dir.mkdir(parents=True)
        dst_dir.mkdir(parents=True)

        stem = "CANON_March_2026"
        _touch_with_mtime(deployment_dir / f"{stem}_2column_cmocean.png", "reprocessed", mtime=2000)
        _touch_with_mtime(dst_dir / f"{stem}_2column_cmocean.png", "stale", mtime=1000)

        with (
            patch("archive.BASE_LRAUV_PATH", base_lrauv_path),
            patch("archive.LRAUV_VOL", str(lrauv_vol)),
        ):
            arch.copy_lrauv_deployment(deployment_dir, stem)

        assert (  # noqa: S101
            dst_dir / f"{stem}_2column_cmocean.png"
        ).read_text() == "reprocessed"

    def test_skips_fresh_index_without_clobber(self, arch, tmp_path):
        base_lrauv_path = tmp_path / "local"
        lrauv_vol = tmp_path / "vol"
        rel = Path("ahi/missionlogs/2026/20260317_20260318")
        deployment_dir = base_lrauv_path / rel
        dst_dir = lrauv_vol / rel
        deployment_dir.mkdir(parents=True)
        dst_dir.mkdir(parents=True)

        stem = "CANON_March_2026"
        _touch_with_mtime(deployment_dir / f"{stem}_2column_cmocean.png", "old", mtime=1000)
        _touch_with_mtime(dst_dir / f"{stem}_2column_cmocean.png", "already archived", mtime=2000)

        with (
            patch("archive.BASE_LRAUV_PATH", base_lrauv_path),
            patch("archive.LRAUV_VOL", str(lrauv_vol)),
        ):
            arch.copy_lrauv_deployment(deployment_dir, stem)

        assert (  # noqa: S101
            dst_dir / f"{stem}_2column_cmocean.png"
        ).read_text() == "already archived"
