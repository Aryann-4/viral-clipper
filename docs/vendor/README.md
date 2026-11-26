# Vendored browser dependencies

The browser app vendors these files so GitHub Pages can serve a static site
without a build step.

| Path | Package | Version | License |
|---|---|---|---|
| `ffmpeg/ffmpeg/*.js` | [`@ffmpeg/ffmpeg`](https://github.com/ffmpegwasm/ffmpeg.wasm) | 0.12.15 | MIT |
| `core/ffmpeg-core.js`, `core/ffmpeg-core.wasm` | [`@ffmpeg/core`](https://github.com/ffmpegwasm/ffmpeg.wasm) | 0.12.10 | GPL-2.0-or-later (includes libx264/libass) |
| `fonts/NotoSans-Bold.ttf` | [Noto Sans](https://github.com/notofonts/noto-fonts) | — | SIL Open Font License 1.1, see `fonts/OFL.txt` |

The speech model and Transformers.js are loaded from jsDelivr/Hugging Face at
runtime and are not vendored here.
