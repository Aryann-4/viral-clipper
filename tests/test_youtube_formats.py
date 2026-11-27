"""Offline regression tests for YouTube stream selection."""

import unittest

from app import _pick_youtube_formats, is_youtube_url


def fmt(**overrides):
    base = {
        "protocol": "https",
        "url": "https://example.invalid/stream",
        "vcodec": "none",
        "acodec": "none",
        "height": None,
        "fps": None,
        "tbr": None,
        "format_id": "x",
        "ext": "mp4",
        "language": None,
        "format_note": "",
    }
    base.update(overrides)
    return base


class YouTubeFormatSelectionTests(unittest.TestCase):
    def test_prefers_original_audio_over_higher_bitrate_dub(self):
        info = {
            "language": "en-US",
            "formats": [
                fmt(format_id="v", vcodec="avc1.640028", height=1080, fps=30, tbr=4000),
                fmt(
                    format_id="a-dub",
                    acodec="mp4a.40.2",
                    tbr=160,
                    language="de-DE",
                    format_note="German (DE), medium",
                ),
                fmt(
                    format_id="a-orig",
                    acodec="mp4a.40.2",
                    tbr=129,
                    language="en-US",
                    format_note="English (US) original (default), medium",
                ),
            ],
        }
        video, audio = _pick_youtube_formats(info)
        self.assertEqual(video["format_id"], "v")
        self.assertEqual(audio["format_id"], "a-orig")

    def test_prefers_1080p_adaptive_over_muxed_360p(self):
        info = {
            "language": "en",
            "formats": [
                fmt(format_id="muxed", vcodec="avc1", acodec="mp4a", height=360, tbr=700),
                fmt(format_id="v1080", vcodec="avc1.640028", height=1080, fps=30, tbr=4000),
                fmt(
                    format_id="a",
                    acodec="mp4a.40.2",
                    tbr=128,
                    language="en",
                    format_note="English original",
                ),
            ],
        }
        video, audio = _pick_youtube_formats(info)
        self.assertEqual(video["format_id"], "v1080")
        self.assertIsNotNone(audio)

    def test_uses_muxed_stream_when_no_standalone_audio(self):
        info = {
            "language": "en",
            "formats": [
                fmt(format_id="muxed", vcodec="avc1", acodec="mp4a", height=720, tbr=1500),
            ],
        }
        video, audio = _pick_youtube_formats(info)
        self.assertEqual(video["format_id"], "muxed")
        self.assertIsNone(audio)

    def test_rejects_video_without_audio(self):
        info = {
            "language": "en",
            "formats": [fmt(format_id="v", vcodec="avc1", height=720)],
        }
        with self.assertRaises(Exception):
            _pick_youtube_formats(info)

    def test_ignores_hls_formats(self):
        info = {
            "language": "en",
            "formats": [
                fmt(
                    format_id="hls",
                    protocol="m3u8_native",
                    url="https://example.invalid/master.m3u8",
                    vcodec="avc1",
                    acodec="mp4a",
                    height=360,
                ),
            ],
        }
        with self.assertRaises(Exception):
            _pick_youtube_formats(info)

    def test_youtube_url_detection(self):
        for value in [
            "https://www.youtube.com/watch?v=abc",
            "https://youtu.be/abc",
            "https://m.youtube.com/watch?v=abc",
            "https://www.youtube.com/shorts/abc",
            "https://www.youtube.com/live/abc",
            "https://music.youtube.com/watch?v=abc",
            "https://www.youtube-nocookie.com/embed/abc",
            "https://www.youtube.com/playlist?list=abc",
        ]:
            self.assertTrue(is_youtube_url(value), value)
        for value in ["/tmp/video.mp4", "https://example.com/watch?v=abc"]:
            self.assertFalse(is_youtube_url(value), value)


if __name__ == "__main__":
    unittest.main()
