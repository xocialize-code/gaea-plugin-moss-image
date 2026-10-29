# gaea-plugin-moss-image

OpenMOSS's **MOSS-Image** (a 5B single-stream DiT on a Qwen3.5-9B text encoder and the FLUX.2 small decoder,
Apache-2.0) as a [Gaea](https://github.com/xocialize/gaea) research plugin. It is built on
[GaeaPluginKit](https://github.com/xocialize/gaea-plugin-kit) over
[moss-image-swift](https://github.com/xocialize/moss-image-swift) 0.3.0, runs in its own process, and speaks the Gaea
Plugin Protocol v1. Add it to Gaea by its repo URL:

```bash
gaea plugins add https://github.com/xocialize/gaea-plugin-moss-image
```

## What it offers

| Model | MES operation | Quality tiers | Notes |
|---|---|---|---|
| `moss-image-sft` ("MOSS-Image") | `image.generate` | standard, best | Long English and Chinese text (line breaks in the prompt are kept), and JSON layout prompts that place text and objects by bounding box (`summary.caption` + `items`, yxyx in pixels of the requested canvas). |
| `moss-image-base` ("MOSS-Image Base") | `image.generate` | best | The pretrained checkpoint SFT is tuned from: the same prompts and layout JSON, less curated composition. For LoRA work and comparison. |

Every offer is **named-only**: it runs only when a request names the model. See [Why named-only](#why-named-only).

- **Sampling.** 50 Euler steps with CFG 3 (two forwards a step): about 2 minutes per 1024² image on an M5 Max.
  Negative prompts are honoured.
- **Sizes.** Sides in multiples of 16, each at least 256 px, at most 1456² in area (the package's
  `MossImageEnvelope`, where its footprints were measured). There is no side cap of its own. A larger area is scaled
  down keeping the aspect, then floored to multiples of 16, and reported `size_snapped`; a shorter side under 256 px
  is refused. The default is 1024². `aspect` with `megapixels` works too.
- **`count`**: 1–8 images, each with `seed + i`. A request without a seed gets one, reported in `effective`.
- **Options**, as MES extensions: `x-moss.steps` (1–100, default 50) and `x-moss.guidance` (1–20, default 3).
- **Not offered:** editing (MOSS-Image is text-to-image only), and LoRA adapters.

## Lanes and memory

Each model has two lanes, each a DiT tier paired with a text-encoder tier, as mlx-image-runtime's catalog pairs them
on a Mac (`ModelCatalog.mossQuant` and `mossTextEncoderQuant`, AB-D-0152). The figures are the package's declared
footprints, as `dist/manifest.json` records them:

| Lane | DiT | Encoder | Resident | Peak while running | Gaea schedules |
|---|---|---|---|---|---|
| `bf16` (default) | bf16 | bf16 (16.8 GB) | 10.5 GB | 16.7 GB | 27.2 GB |
| `int8` | int8 | int8 (9.4 GB) | 5.8 GB | 10.4 GB | 16.2 GB |

The encoder loads for each request and is released before the denoise loop, so the peak is the encode, at every
size up to 1456² (moss-image-swift 0.3.0 decodes the VAE in row bands). In-app measurements: bf16/bf16 26.5 GB,
int8/int8 15.7 GB at 1024² and at the cap. int8 quantizes the bf16 weights on load; its text error is within noise of
bf16 (an OCR gate over 20 English and Chinese long-text prompts, AB-R-0572). The package declares no int4 tier: it
fails the quality gate.

bf16 is the default. When it doesn't fit, Gaea loads the int8 lane and warns (`lane_fallback`, G-23);
`gaea models lane moss-image-sft int8` makes it the default. A request naming a lane (`x-gaea.lane`) never falls back.

## Weights

Gaea sends its models root in `configure` (MLXEngine's store by default, `gaea models root` to change it). The
weights come from the mlx-community mirrors, pinned by commit and byte-identical to OpenMOSS's upstream revisions:

| Repo | Commit | What it gives |
|---|---|---|
| `mlx-community/MOSS-Image-SFT-bf16` | `ea698e99312015505884559ebad94b4a5a0d0753` | SFT's `transformer/`, and the shared `text_encoder/`, `tokenizer/` and `vae/` |
| `mlx-community/MOSS-Image-Base-bf16` | `56e5c1f98640356efce285957df9a883a53064fa` | Base's `transformer/` |

Both models read the Qwen3.5-9B encoder, tokenizer and VAE from the SFT mirror, so a store holding both keeps the
16.8 GB encoder once, and Base needs the SFT mirror's shared files too. The worker downloads a missing snapshot on
`prepare` or `materialize`.

## Build and check locally

```bash
scripts/build-plugin.sh
```

```bash
scripts/check-plugin.sh
```

`check-plugin.sh` generates one 256² image in 8 steps with MOSS-Image int8 (short on purpose) against the weights in
MLXEngine's store (`CHECK_MODELS_ROOT` moves it), with every `.build` folder hidden so MLX can find its metallib only
inside the bundle. A control then shows that the bare worker, outside the bundle, can't. `scripts/release.sh` builds,
checks, notarizes with the fleet's `mlx-notary` profile, and publishes the GitHub release. See the
[demo plugin](https://github.com/xocialize/gaea-plugin-demo) for signing, notarization and CI secrets.

`moss-inproc` (`swift run -c release moss-inproc`; its header lists the options) is not part of the plugin: it runs
the same lane in-process through MLXServeEngine and builds the same request as the worker (the library's `snapped`
and `generateRequest`), the reference that Gaea's runs are compared against byte for byte.

## Why named-only

This is a research plugin (Gaea AB-T-0374), one of a family of opt-in plugins for models the fleet ports but doesn't
default to. The catalog serves MOSS-Image as prod and preferred for nothing: at about 2 minutes per 1024² image it is
for its text rendering and layout prompts, asked for by name. Gaea makes an offer named-only by itself only when the
weights licence isn't permissive, so this plugin sets `"named_only": true` on every offer, and the normalizer never
picks a MOSS model by default.

## Licence

The plugin is Apache-2.0 (Gaea G-21) and ships no weights. MOSS-Image's weights (the DiT, the Qwen3.5-9B encoder
and the FLUX.2 small decoder) are Apache-2.0 (OpenMOSS); shipping was approved on the tech report's data account
(AB-D-0150). moss-image-swift is MIT.
