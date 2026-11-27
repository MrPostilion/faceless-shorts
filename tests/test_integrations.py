"""
Offline tests for faceless_shorts.integrations (unittest, no pytest).

Every test here MUST pass with no network access: no model downloads, no
stock API calls, no LLM calls, no yt-dlp runs against real URLs.
"""

import contextlib
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock

from faceless_shorts import integrations
from faceless_shorts.integrations import (
    ComfyUIClient,
    LLMScriptWriter,
    StockBRoll,
    ToolRegistry,
    WhisperTranscriber,
    YtDlpHelper,
    doctor,
    doctor_report,
)

# The exact tool keys the registry must report.
EXPECTED_TOOLS = ("whisper", "comfyui", "ytdlp", "llm", "stock")
# Every doctor() entry must carry these keys.
INFO_KEYS = ("available", "version", "kind", "install", "hint")
# Env vars that could leak real credentials / endpoints into the tests.
LEAKY_ENV = (
    "PEXELS_API_KEY",
    "PIXABAY_API_KEY",
    "OLLAMA_HOST",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "FACELESS_COMFY_URL",
    "FACELESS_MODELS_DIR",
    "FACELESS_STOCK_DIR",
    "FACELESS_WHISPER_MODEL",
)


@contextlib.contextmanager
def clean_env():
    """Temporarily remove leaky env vars (restored automatically)."""
    with mock.patch.dict(os.environ, {}, clear=False):
        for key in LEAKY_ENV:
            os.environ.pop(key, None)
        yield


def ffmpeg_has_whisper_filter():
    """Independent check: does the local ffmpeg ship the `whisper` filter?"""
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return False
    blob = (proc.stdout or "") + (proc.stderr or "")
    for line in blob.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "whisper":
            return True
    return False


class ToolRegistryTests(unittest.TestCase):
    def test_doctor_shape_and_keys(self):
        with clean_env():
            started = time.monotonic()
            report = doctor()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 6.0, "doctor() must stay fast")
        self.assertIsInstance(report, dict)
        self.assertEqual(set(report.keys()), set(EXPECTED_TOOLS))
        for key in EXPECTED_TOOLS:
            info = report[key]
            self.assertIsInstance(info, dict)
            for required in INFO_KEYS:
                self.assertIn(required, info, f"{key} missing {required}")
            self.assertIsInstance(info["available"], bool)
            self.assertTrue(info["version"] is None or isinstance(info["version"], str))
            self.assertIsInstance(info["kind"], str)
            self.assertTrue(info["kind"])
            self.assertIsInstance(info["install"], str)
            self.assertTrue(info["install"])
            self.assertIsInstance(info["hint"], str)
            self.assertTrue(info["hint"])

    def test_available_returns_bool_for_known_tools(self):
        for key in EXPECTED_TOOLS:
            with self.subTest(tool=key):
                self.assertIsInstance(ToolRegistry.available(key), bool)

    def test_available_unknown_tool_is_false_and_safe(self):
        self.assertIs(ToolRegistry.available("definitely_not_a_tool"), False)
        self.assertIs(ToolRegistry.available(""), False)
        self.assertIs(ToolRegistry.available(None), False)
        self.assertIs(ToolRegistry.available(123), False)


class WhisperTranscriberTests(unittest.TestCase):
    def test_available_is_bool_and_matches_ffmpeg_whisper_filter(self):
        expected = ffmpeg_has_whisper_filter()
        with clean_env():
            started = time.monotonic()
            result = WhisperTranscriber.available()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 6.0)
        self.assertIsInstance(result, bool)
        self.assertEqual(result, expected)

    def test_transcribe_nonexistent_file_returns_none_without_raising(self):
        with clean_env():
            started = time.monotonic()
            result = WhisperTranscriber().transcribe("/nonexistent/faceless_clip.mp4")
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 6.0)
        self.assertIsNone(result)

    def test_transcribe_never_auto_downloads_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = os.path.join(tmp, "models")
            os.makedirs(models_dir)
            media = os.path.join(tmp, "clip.mp4")
            with open(media, "wb") as handle:
                handle.write(b"not a real video")
            with clean_env():
                os.environ["FACELESS_MODELS_DIR"] = models_dir
                with mock.patch.object(WhisperTranscriber, "ensure_model") as ensure:
                    started = time.monotonic()
                    result = WhisperTranscriber().transcribe(media)
                    elapsed = time.monotonic() - started
            # ensure_model must NEVER be invoked automatically...
            ensure.assert_not_called()
            # ...so with no cached model the result is None and nothing was written.
            self.assertIsNone(result)
            self.assertEqual(os.listdir(models_dir), [])
            self.assertLess(elapsed, 6.0)

    def test_ensure_model_returns_cached_path_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = os.path.join(tmp, "ggml-base.en.bin")
            with open(cached, "wb") as handle:
                handle.write(b"fake-model-bytes")
            with clean_env():
                os.environ["FACELESS_MODELS_DIR"] = tmp
                with mock.patch.object(integrations, "_download") as download:
                    result = WhisperTranscriber.ensure_model("en")
            self.assertEqual(result, cached)
            download.assert_not_called()
            self.assertFalse(os.path.exists(cached + ".part"))

    def test_ensure_model_honours_requested_size_over_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = os.path.join(tmp, "ggml-small.bin")
            with open(cached, "wb") as handle:
                handle.write(b"fake-model-bytes")
            with clean_env():
                os.environ["FACELESS_MODELS_DIR"] = tmp
                os.environ["FACELESS_WHISPER_MODEL"] = "tiny"
                with mock.patch.object(integrations, "_download") as download:
                    result = WhisperTranscriber.ensure_model("auto", model_size="small")
            self.assertEqual(result, cached)
            download.assert_not_called()

    def test_model_filename_language_rules(self):
        name = WhisperTranscriber._model_filename
        self.assertEqual(name("base", "en"), "ggml-base.en.bin")
        self.assertEqual(name("base", "en-US"), "ggml-base.en.bin")
        for language in ("auto", None, "", "sn", "hi"):
            with self.subTest(language=language):
                self.assertEqual(name("base", language), "ggml-base.bin")
        # whisper.cpp ships no "ggml-large.bin" and no English-only large model.
        self.assertEqual(name("large", "en"), "ggml-large-v3.bin")
        self.assertEqual(name("large", "auto"), "ggml-large-v3.bin")

    def test_find_model_never_uses_english_only_model_for_other_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            english_only = os.path.join(tmp, "ggml-base.en.bin")
            with open(english_only, "wb") as handle:
                handle.write(b"fake-model-bytes")
            with clean_env():
                os.environ["FACELESS_MODELS_DIR"] = tmp
                self.assertEqual(WhisperTranscriber._find_model("base", "en"), english_only)
                self.assertIsNone(WhisperTranscriber._find_model("base", "auto"))
                self.assertIsNone(WhisperTranscriber._find_model("base", None))
                multilingual = os.path.join(tmp, "ggml-base.bin")
                with open(multilingual, "wb") as handle:
                    handle.write(b"fake-model-bytes")
                self.assertEqual(WhisperTranscriber._find_model("base", "auto"), multilingual)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_filter_value_escaping_survives_ffmpeg_parsing(self):
        # ametadata's file= option takes a path, so ffmpeg itself proves the
        # escaping: the file only appears if the path arrived intact. The temp
        # dir supplies "C:\..." on Windows; POSIX also gets a literal ":" and "\".
        name = "it's [a],b;c.txt" if os.name == "nt" else "it's [a],b;c:d\\e.txt"
        with tempfile.TemporaryDirectory(prefix="fs esc ") as tmp:
            target = os.path.join(tmp, name)
            proc = subprocess.run(
                [
                    shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "anullsrc", "-t", "0.1",
                    "-af", "ametadata=mode=print:file=" + integrations._ffmpeg_filter_value(target),
                    "-f", "null", "-",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(os.path.exists(target), f"ffmpeg mangled {target!r}")

    def test_parse_whisper_json_handles_unescaped_quotes(self):
        # Verbatim shape of FFmpeg 9.1 whisper output: JSON Lines, quotes in
        # "text" left unescaped (the second line is not valid JSON).
        payload = (
            '{"start":0,"end":2960,"text":"So guys I\'m going to tell a story","language":"en"}\n'
            '{"start":2994,"end":7760,"text":"asking, "Why is this person back?"","language":"en"}\n'
            "\n"
        )
        self.assertEqual(
            integrations._parse_whisper_json(payload),
            [
                {"start": 0, "end": 2960, "text": "So guys I'm going to tell a story", "language": "en"},
                {"start": 2994, "end": 7760, "text": 'asking, "Why is this person back?"', "language": "en"},
            ],
        )
        self.assertIsNone(integrations._parse_whisper_json("not json\n"))

    def _run_fake_transcribe(self, out_format, payload, **kwargs):
        """Run transcribe() with ffmpeg mocked to write *payload*; return
        (result, ffmpeg argv, subprocess.run kwargs, destination path)."""
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            media = os.path.join(tmp, "clip.mp4")
            with open(media, "wb") as handle:
                handle.write(b"not a real video")
            model = os.path.join(tmp, "models", "ggml-base.bin")
            destination = os.path.join(tmp, f"faceless_whisper_fixed.{out_format}")

            def fake_run(cmd, **run_kwargs):
                calls.append((cmd, run_kwargs))
                with open(destination, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                return subprocess.CompletedProcess(cmd, 0, "", "")

            with mock.patch.object(WhisperTranscriber, "available", return_value=True), \
                    mock.patch.object(WhisperTranscriber, "_find_model", return_value=model), \
                    mock.patch.object(integrations, "_ffmpeg_path", return_value="ffmpeg"), \
                    mock.patch.object(integrations.tempfile, "gettempdir", return_value=tmp), \
                    mock.patch.object(integrations.uuid, "uuid4", return_value=mock.Mock(hex="fixed")), \
                    mock.patch.object(integrations.subprocess, "run", side_effect=fake_run):
                result = WhisperTranscriber.transcribe(media, out_format=out_format, **kwargs)
            self.assertEqual(len(calls), 1)
            self.assertFalse(os.path.exists(destination), "temp output must be cleaned up")
            cmd, run_kwargs = calls[0]
            return result, cmd, run_kwargs, model, destination

    def test_transcribe_escapes_paths_and_parses_json_lines(self):
        payload = '{"start":0,"end":1500,"text":"He said "hi"","language":"en"}\n'
        result, cmd, run_kwargs, model, destination = self._run_fake_transcribe(
            "json", payload, language="auto", timeout=None
        )
        self.assertEqual(result, [{"start": 0, "end": 1500, "text": 'He said "hi"', "language": "en"}])
        filter_expr = cmd[cmd.index("-af") + 1]
        self.assertIn("model=" + integrations._ffmpeg_filter_value(model), filter_expr)
        self.assertIn("destination=" + integrations._ffmpeg_filter_value(destination), filter_expr)
        self.assertIn(":language=auto", filter_expr)
        self.assertIsNone(run_kwargs["timeout"])

    def test_transcribe_srt_returns_file_text_with_blank_lines(self):
        payload = "0\n00:00:00,000 --> 00:00:01,500\nHello\n\n1\n00:00:01,500 --> 00:00:03,000\nWorld\n\n"
        result, _, run_kwargs, _, _ = self._run_fake_transcribe("srt", payload)
        self.assertEqual(result, payload)
        self.assertEqual(run_kwargs["timeout"], integrations.TRANSCRIBE_TIMEOUT)


class ComfyUIClientTests(unittest.TestCase):
    def test_available_false_fast_when_nothing_listens(self):
        with clean_env():
            client = ComfyUIClient()
            started = time.monotonic()
            result = client.available()
            elapsed = time.monotonic() - started
        self.assertIs(result, False)
        self.assertLess(elapsed, 3.0, "availability probe must be fast")

    def test_generate_on_unavailable_client_returns_none_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "render.png")
            with clean_env():
                client = ComfyUIClient()
                started = time.monotonic()
                result = client.generate("a serene mountain lake at dawn", target)
                elapsed = time.monotonic() - started
            self.assertIsNone(result)
            self.assertFalse(os.path.exists(target))
            self.assertLess(elapsed, 6.0)


class YtDlpHelperTests(unittest.TestCase):
    def test_available_returns_bool_quickly(self):
        with clean_env():
            started = time.monotonic()
            result = YtDlpHelper.available()
            elapsed = time.monotonic() - started
        self.assertIsInstance(result, bool)
        self.assertLess(elapsed, 6.0)

    def test_download_returns_none_when_unavailable(self):
        if YtDlpHelper.available():
            self.skipTest("yt-dlp is installed here; download() would hit the network")
        with tempfile.TemporaryDirectory() as tmp:
            with clean_env():
                started = time.monotonic()
                result = YtDlpHelper.download(
                    "https://example.com/reference.mp4", out_dir=tmp
                )
                elapsed = time.monotonic() - started
            self.assertIsNone(result)
            self.assertEqual(os.listdir(tmp), [])
            self.assertLess(elapsed, 6.0)


class LLMScriptWriterTests(unittest.TestCase):
    def test_available_returns_bool_quickly(self):
        with clean_env():
            started = time.monotonic()
            result = LLMScriptWriter.available()
            elapsed = time.monotonic() - started
        self.assertIsInstance(result, bool)
        self.assertLess(elapsed, 6.0)

    def test_write_script_returns_none_offline(self):
        with clean_env():
            # Point Ollama at a dead local port and remove every LLM env var
            # so no backend can possibly be reached.
            os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"
            writer = LLMScriptWriter()
            started = time.monotonic()
            self.assertFalse(writer.available())
            result = writer.write_script("stoic discipline", language="en")
            elapsed = time.monotonic() - started
        self.assertIsNone(result)
        self.assertLess(elapsed, 6.0)


class StockBRollTests(unittest.TestCase):
    def test_unavailable_without_keys_and_search_returns_none(self):
        with clean_env():
            self.assertFalse(StockBRoll.available())
            started = time.monotonic()
            result = StockBRoll.search_clip("ocean waves", orientation="portrait")
            elapsed = time.monotonic() - started
        self.assertIsNone(result)
        self.assertLess(elapsed, 6.0)


class DoctorReportTests(unittest.TestCase):
    def test_doctor_report_contains_every_tool_key(self):
        with clean_env():
            started = time.monotonic()
            text = doctor_report()
            elapsed = time.monotonic() - started
        self.assertLess(elapsed, 6.0)
        self.assertIsInstance(text, str)
        self.assertTrue(text.strip())
        for key in EXPECTED_TOOLS:
            self.assertIn(key, text)

    def test_all_exports(self):
        expected = {
            "ToolRegistry",
            "WhisperTranscriber",
            "ComfyUIClient",
            "YtDlpHelper",
            "LLMScriptWriter",
            "StockBRoll",
            "doctor",
            "doctor_report",
        }
        self.assertEqual(set(integrations.__all__), expected)
        for name in expected:
            self.assertTrue(hasattr(integrations, name), f"missing export {name}")


if __name__ == "__main__":
    unittest.main()
