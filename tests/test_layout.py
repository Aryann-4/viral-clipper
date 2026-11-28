"""Offline tests for the split (streamer top / gameplay bottom) layout."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import app


class FacecamCropParsingTests(unittest.TestCase):
    def test_parses_normalized_tuple(self):
        self.assertEqual(
            app.parse_facecam_crop("0.70, 0.55, 0.28, 0.40"),
            (0.70, 0.55, 0.28, 0.40),
        )

    def test_empty_values_mean_auto_detect(self):
        self.assertIsNone(app.parse_facecam_crop(""))
        self.assertIsNone(app.parse_facecam_crop(None))

    def test_rejects_out_of_bounds_crop(self):
        with self.assertRaises(ValueError):
            app.parse_facecam_crop("0.8,0.8,0.3,0.3")
        with self.assertRaises(ValueError):
            app.parse_facecam_crop("0,0,0,0.4")
        with self.assertRaises(ValueError):
            app.parse_facecam_crop("left,top,right,bottom")


class FacecamPixelTests(unittest.TestCase):
    def test_outputs_even_in_bounds_pixels(self):
        media = SimpleNamespace(width=1920, height=1080)
        x, y, w, h = app._facecam_pixels(media, (0.71, 0.53, 0.29, 0.45))
        self.assertEqual((x % 2, y % 2, w % 2, h % 2), (0, 0, 0, 0))
        self.assertLessEqual(x + w, media.width)
        self.assertLessEqual(y + h, media.height)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_clamps_oversized_crop(self):
        media = SimpleNamespace(width=640, height=480)
        x, y, w, h = app._facecam_pixels(media, (0.9, 0.9, 1.0, 1.0))
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + w, media.width)
        self.assertLessEqual(y + h, media.height)


class SplitFilterTests(unittest.TestCase):
    def test_split_script_stacks_webcam_over_gameplay(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            script = app._write_filter_script(
                workdir,
                None,
                "0",
                layout="split",
                facecam_pixels=(1280, 720, 480, 360),
            )
            text = script.read_text(encoding="utf-8")
            self.assertIn("split=2[top][bot]", text)
            self.assertIn("crop=w=480:h=360:x=1280:y=720", text)
            self.assertIn(
                f"vstack=inputs=2,fps={app.OUTPUT_FPS}", text
            )
            self.assertIn(f"crop={app.TARGET_WIDTH}:{app.SPLIT_TOP_HEIGHT}", text)
            self.assertIn(
                f"crop={app.TARGET_WIDTH}:{app.SPLIT_BOTTOM_HEIGHT}", text
            )

    def test_split_requires_a_webcam_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(app.ClipperError):
                app._write_filter_script(
                    Path(tmp), None, "0", layout="split", facecam_pixels=None
                )

    def test_full_layout_uses_single_vertical_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = app._write_filter_script(
                Path(tmp), None, "max(0,(in_w-1080)/2)", layout="full"
            )
            text = script.read_text(encoding="utf-8")
            self.assertIn("scale=1080:1920", text)
            self.assertNotIn("vstack", text)


class SplitCaptionTests(unittest.TestCase):
    def test_seam_captions_are_positioned(self):
        words = [app.Word(0.2, 0.8, "HELLO"), app.Word(0.9, 1.4, "WORLD")]
        with tempfile.TemporaryDirectory() as tmp:
            path = app.build_ass_captions(
                words, 0.0, 2.0, Path(tmp) / "captions.ass",
                caption_y=app.SPLIT_CAPTION_Y,
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn(f"\\pos({app.TARGET_WIDTH // 2},{app.SPLIT_CAPTION_Y})", text)
            self.assertIn(",5,70,70,0,1", text)

    def test_default_captions_use_bottom_margin(self):
        words = [app.Word(0.2, 0.8, "HELLO")]
        with tempfile.TemporaryDirectory() as tmp:
            path = app.build_ass_captions(
                words, 0.0, 2.0, Path(tmp) / "captions.ass"
            )
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("\\pos(", text)
            self.assertIn(",2,70,70,330,1", text)


if __name__ == "__main__":
    unittest.main()
