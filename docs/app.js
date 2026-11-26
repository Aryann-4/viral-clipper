import { FFmpeg } from "./vendor/ffmpeg/ffmpeg/index.js";
import {
  normalizeWords,
  sentenceUnits,
  generateCandidates,
  energyEnvelope,
  scoreCandidates,
  selectClips,
  buildAss,
} from "./core.js";

const $ = (selector) => document.querySelector(selector);
const url = (path) => new URL(path, import.meta.url).href;

const fileInput = $("#file");
const modelSelect = $("#model");
const durationSelect = $("#duration");
const clipsSelect = $("#clips");
const runButton = $("#run");
const status = $("#status");
const progressBar = $("#progress-bar");
const transcriptBox = $("#transcript");
const results = $("#results");
const preview = $("#preview");
const errorBox = $("#error");
const dropZone = $("#drop-zone");

let ffmpeg = null;
let transcriber = null;
let loadedModel = "";
let lastLog = "";
let outputUrls = [];
let previewUrl = "";
let inputName = "input.mp4";
let words = [];
const tempFiles = new Set();

function trackTemp(name) {
  tempFiles.add(name);
}

async function cleanupTempFiles() {
  if (!ffmpeg) return;
  for (const name of [...tempFiles]) {
    try {
      await ffmpeg.deleteFile(name);
    } catch {
      // Already removed or never created.
    }
    tempFiles.delete(name);
  }
}

function setStatus(message) {
  status.textContent = message;
}

function setProgress(value) {
  progressBar.style.width = `${Math.max(0, Math.min(100, value))}%`;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}

function formatTime(seconds) {
  const value = Math.max(0, Math.round(seconds));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

function setPreview(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
}

function clearResults() {
  for (const value of outputUrls) URL.revokeObjectURL(value);
  outputUrls = [];
  results.replaceChildren();
  transcriptBox.textContent = "";
}

async function loadFFmpeg() {
  if (ffmpeg) return ffmpeg;
  setStatus("Loading FFmpeg engine (about 31 MB)…");
  setProgress(4);
  ffmpeg = new FFmpeg();
  ffmpeg.on("log", ({ message }) => {
    lastLog = message || lastLog;
  });
  ffmpeg.on("progress", ({ progress }) => {
    if (Number.isFinite(progress)) {
      setProgress(68 + Math.min(1, progress) * 22);
    }
  });
  await ffmpeg.load({
    coreURL: url("./vendor/ffmpeg/core/ffmpeg-core.js"),
    wasmURL: url("./vendor/ffmpeg/core/ffmpeg-core.wasm"),
  });
  const font = await fetch(url("./vendor/ffmpeg/fonts/NotoSans-Bold.ttf")).then((response) => {
    if (!response.ok) throw new Error("Could not load the bundled caption font.");
    return response.arrayBuffer();
  });
  await ffmpeg.writeFile("arial.ttf", new Uint8Array(font));
  return ffmpeg;
}

async function loadTranscriber(modelId) {
  if (transcriber && loadedModel === modelId) return transcriber;
  setStatus(`Loading speech model ${modelId}…`);
  setProgress(18);
  const { pipeline, env } = await import(
    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/+esm"
  );
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  transcriber = await pipeline("automatic-speech-recognition", modelId, {
    device: "wasm",
    progress_callback: (event) => {
      if (event?.status === "progress" && Number.isFinite(event.progress)) {
        setStatus(`Downloading speech model… ${Math.round(event.progress)}%`);
        setProgress(18 + (event.progress / 100) * 17);
      }
    },
  });
  loadedModel = modelId;
  return transcriber;
}

async function extractAudio(file) {
  setStatus("Extracting audio for transcription…");
  setProgress(35);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const extension = file.name.toLowerCase().match(/\.(mp4|mkv|mov|webm|m4v|avi)$/)?.[1] || "mp4";
  inputName = `input.${extension}`;
  trackTemp(inputName);
  await ffmpeg.writeFile(inputName, bytes);
  lastLog = "";
  const code = await ffmpeg.exec([
    "-i", inputName,
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    "audio.wav",
  ]);
  if (code !== 0) {
    throw new Error(`FFmpeg could not read the audio track. ${lastLog.slice(-400)}`);
  }
  const wav = await ffmpeg.readFile("audio.wav");
  trackTemp("audio.wav");
  return new Blob([wav], { type: "audio/wav" }).arrayBuffer();
}

async function decodeSamples(arrayBuffer) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const context = new AudioContextClass();
  try {
    // decodeAudioData detaches its input buffer, so decode a copy to keep the WAV intact.
    const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
    return { samples: buffer.getChannelData(0), sampleRate: buffer.sampleRate };
  } finally {
    await context.close();
  }
}

async function transcribeAudio(blob) {
  const audioUrl = URL.createObjectURL(blob);
  try {
    setStatus("Transcribing speech in your browser…");
    setProgress(45);
    const result = await transcriber(audioUrl, {
      return_timestamps: "word",
      chunk_length_s: 30,
      stride_length_s: 5,
    });
    return normalizeWords(result?.chunks || []);
  } finally {
    URL.revokeObjectURL(audioUrl);
  }
}

async function renderClip(clip, index) {
  const clipStart = Math.max(0, clip.start - 0.10);
  const clipEnd = clip.end + 0.18;
  const duration = clipEnd - clipStart;
  const captionName = `captions_${index}.ass`;
  const outputName = `clip_${index}.mp4`;
  const ass = buildAss(words, clipStart, duration);
  trackTemp(captionName);
  trackTemp(outputName);
  await ffmpeg.writeFile(captionName, new TextEncoder().encode(ass));

  const filter = [
    "scale=1080:1920:force_original_aspect_ratio=increase",
    "setsar=1",
    "crop=1080:1920",
    "fps=30",
    `ass=filename=/${captionName}:fontsdir=/`,
  ].join(",");

  setStatus(`Rendering clip ${index + 1}…`);
  setProgress(68 + index * 8);
  lastLog = "";
  const code = await ffmpeg.exec([
    "-y",
    "-ss", clipStart.toFixed(3),
    "-i", inputName,
    "-t", duration.toFixed(3),
    "-map", "0:v:0",
    "-map", "0:a?",
    "-vf", filter,
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-crf", "24",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "128k",
    "-movflags", "+faststart",
    outputName,
  ]);
  if (code !== 0) {
    throw new Error(`Clip ${index + 1} could not be rendered. ${lastLog.slice(-400)}`);
  }
  const data = await ffmpeg.readFile(outputName);
  const blob = new Blob([data], { type: "video/mp4" });
  const videoUrl = URL.createObjectURL(blob);
  outputUrls.push(videoUrl);
  return { ...clip, clipStart, clipEnd, duration, videoUrl, blob };
}

function showTranscript(clips) {
  const lines = clips.map((clip) => {
    return `#${clip.rank} · ${formatTime(clip.clipStart)}–${formatTime(clip.clipEnd)} · score ${clip.score}\n${clip.text}`;
  });
  transcriptBox.textContent = lines.join("\n\n");
}

function showResults(clips) {
  for (const clip of clips) {
    const card = document.createElement("article");
    card.className = "result-card";

    const video = document.createElement("video");
    video.src = clip.videoUrl;
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";

    const meta = document.createElement("div");
    meta.className = "result-meta";
    meta.textContent = `Clip ${clip.rank} · ${formatTime(clip.clipStart)}–${formatTime(clip.clipEnd)} · ${clip.score} pts`;

    const download = document.createElement("a");
    download.className = "download";
    download.href = clip.videoUrl;
    download.download = `viral-clip-${clip.rank}.mp4`;
    download.textContent = "Download MP4 ↗";

    card.append(video, meta, download);
    results.append(card);
  }
}

async function run() {
  clearError();
  clearResults();
  const file = fileInput.files?.[0];
  if (!file) {
    showError("Choose an MP4, MKV, MOV, or WebM video first.");
    return;
  }
  if (file.size > 600 * 1024 * 1024) {
    showError("This browser demo is safest with files under 600 MB. Use the local app for larger videos.");
    return;
  }

  runButton.disabled = true;
  runButton.textContent = "Working…";
  try {
    setPreview(file);
    await loadFFmpeg();
    await loadTranscriber(modelSelect.value);
    const wavBuffer = await extractAudio(file);
    const { samples, sampleRate } = await decodeSamples(wavBuffer);
    words = await transcribeAudio(new Blob([wavBuffer], { type: "audio/wav" }));
    if (words.length < 5) {
      throw new Error("No speech was detected. Try a video with clearer spoken audio.");
    }

    setStatus("Finding the strongest sentence-aligned moments…");
    setProgress(62);
    const target = Number(durationSelect.value);
    const clipCount = Number(clipsSelect.value);
    const totalDuration = words[words.length - 1].end;
    const sentences = sentenceUnits(words);
    const envelope = energyEnvelope(samples, sampleRate);
    const candidates = scoreCandidates(
      generateCandidates(sentences, target, totalDuration),
      envelope,
      target
    );
    if (!candidates.length) {
      throw new Error("No usable highlight could be found in this transcript.");
    }
    const selected = selectClips(candidates, clipCount);
    const rendered = [];
    for (let index = 0; index < selected.length; index += 1) {
      rendered.push(await renderClip(selected[index], index));
    }

    showTranscript(rendered);
    showResults(rendered);
    setProgress(100);
    setStatus(`Done — ${rendered.length} captioned vertical clip${rendered.length === 1 ? "" : "s"} ready.`);
  } catch (error) {
    console.error(error);
    showError(error?.message || String(error));
    setStatus("Failed. See the message above.");
  } finally {
    await cleanupTempFiles();
    runButton.disabled = false;
    runButton.textContent = "Generate clips in browser";
  }
}

runButton.addEventListener("click", run);

fileInput.addEventListener("change", () => {
  clearError();
  const file = fileInput.files?.[0];
  if (!file) return;
  setPreview(file);
  setStatus(`${file.name} ready. Choose settings and generate.`);
});

for (const [eventName] of [["dragenter"], ["dragover"]]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
}
for (const [eventName] of [["dragleave"], ["drop"]]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
}
dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  fileInput.files = event.dataTransfer.files;
  fileInput.dispatchEvent(new Event("change"));
});
