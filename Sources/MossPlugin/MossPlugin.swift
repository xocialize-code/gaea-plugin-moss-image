import Foundation
import GaeaPluginKit
import GaeaProtocol
import MLXMossImage
import MLXToolKit
import MossImage

// gaea-plugin-moss-image: OpenMOSS's MOSS-Image (Apache-2.0 weights, MIT port) as a Gaea research plugin, on
// moss-image-swift through MLXServeEngine (Gaea AB-T-0374).
//
// Two checkpoints, as mlx-image-runtime's catalog serves them (`ModelCatalog.moss`):
// - moss-image-sft ("MOSS-Image"): a 5B single-stream DiT on a Qwen3.5-9B encoder, with the strongest open-weight
//   long-text rendering in English and Chinese, and JSON layout prompts that place text and objects by bounding box;
// - moss-image-base ("MOSS-Image Base"): the pretrained checkpoint SFT is tuned from, for LoRA work and comparison.
// Both are text-to-image only: 50 Euler steps with CFG 3, so negative prompts count.
//
// Every offer is named-only: the normalizer never picks a MOSS model by default, and a request runs one only by naming
// it. Each model has two lanes, one per DiT + encoder tier pair, the catalog's Mac ladder (`mossQuant`,
// `mossTextEncoderQuant`): bf16/bf16 from 48 GB, int8/int8 below it. Weights come from the models root Gaea sends in
// configure; the engine materializes a missing snapshot. Both models read the text encoder, tokenizer and VAE from the
// SFT mirror, so a store holding both keeps the 16.8 GB encoder once.

/// The package's envelope (`MossImageEnvelope`), the one its footprints were measured at: sides in multiples of 16,
/// each at least 256 px, at most 1456² in area. No side cap of its own: the area and the shorter side bound it.
public let maxPixels = MossImageEnvelope.maxOutputPixels
public let minSide = MossImageEnvelope.minSide, grid = 16

public enum Model: String, CaseIterable, Sendable {
    case sft = "moss-image-sft"
    case base = "moss-image-base"

    public var variant: MossImageVariant { self == .sft ? .sft : .base }
    public var title: String { self == .sft ? "MOSS-Image" : "MOSS-Image Base" }

    /// The catalog's defaults for both checkpoints (`GenerationDefaults(steps: 50, guidance: 3.0)`).
    public var defaultSteps: Int { 50 }
    public var defaultGuidance: Double { 3 }

    /// The package configuration for one lane; moss-inproc builds the same one. Empty snapshot paths: the engine's store
    /// resolves this variant's DiT, and the SFT mirror's shared encoder and VAE.
    public func configuration(lane: Lane) -> MossImageConfiguration {
        MossImageConfiguration(variant: variant, quant: lane.quant, snapshotPath: "", sharedSnapshotPath: "",
                               defaultSteps: defaultSteps, defaultGuidanceScale: defaultGuidance,
                               textEncoderQuant: lane.textEncoderQuant)
    }
}

/// A lane is a DiT tier and an encoder tier together, as the catalog pairs them. The package declares bf16 and int8
/// DiTs only (int4 fails its quality gate); an int8 encoder quantizes the Qwen3.5-9B Linears on load (9.4 GB, not 16.8).
public enum Lane: String, CaseIterable, Sendable {
    case bf16, int8

    public var quant: Quant { self == .bf16 ? .bf16 : .int8 }
    public var textEncoderQuant: MossImageTextEncoderQuant { self == .bf16 ? .bf16 : .int8 }
}

/// The MES size a job asks for (`size`, or `aspect` with `megapixels`, Image profile §2), else `fallback`.
func requestedSize(_ job: Job, fallback: (Int, Int)) throws -> (Int, Int) {
    if let size = job.param("size") {
        guard let w = size["width"]?.intValue, let h = size["height"]?.intValue, w > 0, h > 0 else {
            throw GPPError(.invalidRequest, "size is {width, height} in pixels.")
        }
        return (w, h)
    }
    if let aspect = job.param("aspect")?.stringValue {
        let parts = aspect.split(separator: ":").compactMap { Double($0) }
        guard parts.count == 2, parts[0] > 0, parts[1] > 0 else { throw GPPError(.invalidRequest, "aspect is W:H, e.g. 16:9.") }
        let pixels = (job.param("megapixels")?.doubleValue ?? 1.0) * 1_000_000
        let height = (pixels * parts[1] / parts[0]).squareRoot()
        return (Int(height * parts[0] / parts[1]), Int(height))
    }
    return fallback
}

/// Fits a size to MOSS-Image's envelope: scaled down to the area cap, keeping the aspect, then floored to multiples of
/// 16. A change is reported as `size_snapped`; a shorter side under 256 px is refused. moss-inproc snaps the same way.
public func snapped(_ size: (Int, Int)) throws -> (width: Int, height: Int, warning: Warning?) {
    var (w, h) = (Double(size.0), Double(size.1))
    if w * h > Double(maxPixels) {
        let scale = (Double(maxPixels) / (w * h)).squareRoot()
        w *= scale
        h *= scale
    }
    let width = Int(w) / grid * grid, height = Int(h) / grid * grid
    guard min(width, height) >= minSide else {
        throw GPPError(.invalidRequest, "MOSS-Image draws at least \(minSide) px a side and at most 1456² in area; "
            + "\(size.0)×\(size.1) doesn't fit.", detail: ["param": "size"])
    }
    let warning = (width, height) == size ? nil : Warning(
        code: "size_snapped", message: "Drew \(width)×\(height) for \(size.0)×\(size.1): multiples of 16, at most 1456² in area.",
        param: "size")
    return (width, height, warning)
}

/// The engine request for one image, the one place it is built: the worker sends it for image `i` with `seed + i`, and
/// moss-inproc sends it for its seed. An empty negative prompt is none (the package encodes "" for it either way).
public func generateRequest(prompt: String, negative: String?, width: Int, height: Int, steps: Int, guidance: Double,
                            seed: UInt64) -> T2IRequest {
    T2IRequest(prompt: prompt, negativePrompt: negative.flatMap { $0.isEmpty ? nil : $0 }, width: width, height: height,
               steps: steps, guidanceScale: guidance, seed: seed)
}

/// The run's settings: steps, guidance, seed, count, and the warnings for what the plugin ignores.
struct Shared {
    var steps: Int
    var guidance: Double
    var negative: String?
    var seed: UInt64
    var count: Int
    var warnings: [Warning] = []

    init(_ job: Job, model: Model) throws {
        steps = job.param("x-moss.steps")?.intValue ?? model.defaultSteps
        guard (1...100).contains(steps) else { throw GPPError(.invalidRequest, "x-moss.steps is 1–100.") }
        guidance = job.param("x-moss.guidance")?.doubleValue ?? model.defaultGuidance
        guard (1...20).contains(guidance) else { throw GPPError(.invalidRequest, "x-moss.guidance is 1–20.") }
        count = job.param("count")?.intValue ?? 1
        guard (1...8).contains(count) else { throw GPPError(.invalidRequest, "count is 1–8.") }
        // A CFG model: the negative prompt is the second branch's text.
        negative = job.param("negative_prompt")?.stringValue.flatMap { $0.isEmpty ? nil : $0 }
        if job.param("prompt_enhance")?.stringValue == "on" {
            warnings.append(Warning(code: "prompt_enhance_ignored", message: "This plugin doesn't rewrite prompts; Gaea's enhancer does.",
                                    param: "prompt_enhance"))
        }
        // Image profile §2: a request without a seed still gets one, reported in effective.
        seed = job.seed ?? UInt64.random(in: 0..<(1 << 31))
    }

    func effective(width: Int, height: Int) -> [String: JSONValue] {
        var values: [String: JSONValue] = [
            "size": ["width": .int(Int64(width)), "height": .int(Int64(height))],
            "seed": .int(Int64(seed)), "count": .int(Int64(count)),
            "x-moss.steps": .int(Int64(steps)), "x-moss.guidance": .double(guidance),
        ]
        if let negative { values["negative_prompt"] = .string(negative) }
        return values
    }
}

/// MES image.generate: one engine call per image, each with seed + i. A JSON layout prompt is just the prompt text.
func generate(_ model: Model) -> EngineOperation {
    EngineOperation("image.generate", outputNames: ["image": "images"]) { job in
        guard let prompt = try job.text("prompt"), !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw GPPError(.invalidRequest, "image.generate needs a prompt.")
        }
        let size = try snapped(try requestedSize(job, fallback: (1024, 1024)))
        var shared = try Shared(job, model: model)
        if let warning = size.warning { shared.warnings.insert(warning, at: 0) }
        let effective = shared.effective(width: size.width, height: size.height)
        return (0..<shared.count).map { i in
            EngineCall(generateRequest(prompt: prompt, negative: shared.negative, width: size.width, height: size.height,
                                       steps: shared.steps, guidance: shared.guidance, seed: shared.seed + UInt64(i)),
                       effective: effective, warnings: shared.warnings)
        }
    }
}

public func lanes(_ model: Model) -> [LaneDefinition] {
    Lane.allCases.map { lane in
        LaneDefinition(id: lane.rawValue, default: lane == .bf16, configuration: model.configuration(lane: lane))
    }
}

let apache: JSONValue = ["spdx": "Apache-2.0", "commercial_use": true,
                         "terms_url": "https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT"]
let limits = PluginManifest.Limits(maxPixels: maxPixels, minSide: minSide, sizeMultiple: grid)

func params(_ model: Model) -> JSONValue {
    let properties: [String: JSONValue] = [
        "x-moss.steps": ["type": "integer", "minimum": 1, "maximum": 100, "default": .int(Int64(model.defaultSteps)),
                         "description": "Euler denoising steps; more is slower and finer. Each step is two forwards (CFG).",
                         "x-mes-ui": ["widget": "slider", "label": "Steps", "advanced": true]],
        "x-moss.guidance": ["type": "number", "minimum": 1, "maximum": 20, "default": .double(model.defaultGuidance),
                            "description": "Classifier-free guidance: how strongly the prompt (and against the negative prompt) steers.",
                            "x-mes-ui": ["widget": "slider", "label": "Guidance", "advanced": true]],
    ]
    return ["type": "object", "properties": .object(properties), "additionalProperties": false]
}

func offerModel(_ model: Model) -> JSONValue {
    let notes: String = switch model {
    case .sft: "Long English and Chinese text (line breaks kept), and JSON layout prompts that place text and objects by "
        + "bounding box (summary.caption + items, yxyx in pixels of the requested canvas). 50 steps with CFG 3: about 2 minutes "
        + "per 1024² image on an M5 Max. Sides multiples of 16, 256²–1456² by area."
    case .base: "The pretrained checkpoint MOSS-Image is tuned from: the same prompts and layout JSON, less curated "
        + "composition; for LoRA work and comparison. 50 steps with CFG 3: about 2 minutes per 1024² image on an M5 Max."
    }
    return [
        "id": .string(model.rawValue), "title": .string(model.title), "license": apache,
        "memory_class": "large", "speed_class": "slow",
        "quality_tiers": model == .sft ? ["standard", "best"] : ["best"], "named_only": true,
        "notes": .string("Research plugin, named-only: runs only when a request names it. " + notes),
    ]
}

public let mossPlugin = GaeaPluginDefinition(
    id: "com.xocialize.gaea.moss-image",
    version: "0.1.0",
    minGaea: "0.2.0",
    team: "CFZS6RXUCP",
    repo: "https://github.com/xocialize/gaea-plugin-moss-image",
    title: "MOSS-Image",
    summary: "OpenMOSS's MOSS-Image (Apache-2.0): a 5B single-stream DiT on a Qwen3.5-9B encoder and the FLUX.2 small "
        + "decoder. Text-to-image with long English and Chinese text and JSON layout prompts (SFT), and the pretrained "
        + "Base. A research plugin: every model is named-only.",
    license: "Apache-2.0",
    executable: "gaea-moss-image-worker",
    models: Model.allCases.map { model in
        model == .sft
            ? .engine(MossImageSFTPackage.self, id: model.rawValue, title: model.title, lanes: lanes(model),
                      operations: [generate(model)])
            : .engine(MossImageBasePackage.self, id: model.rawValue, title: model.title, lanes: lanes(model),
                      operations: [generate(model)])
    },
    offers: Model.allCases.map { model in
        PluginManifest.Offer(operation: "image.generate", profile: "image", model: offerModel(model), params: params(model),
                             defaults: ["count": 1], limits: limits)
    })
