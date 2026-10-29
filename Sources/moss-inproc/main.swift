import Foundation
import MLXMossImage
import MLXServeCore
import MLXToolKit
import MossPlugin

// The in-process reference for the plugin: one MOSS-Image lane loaded by MLXServeEngine in this process, with the lane
// configuration the worker uses (MossPlugin) and the request it would build (`snapped`, `generateRequest`). A cold run
// then a warm one; the first image is written to --out, and one JSON line of timings is printed. Not part of the plugin
// bundle.
//
//   moss-inproc [--model moss-image-sft | moss-image-base] [--lane bf16 | int8] [--prompt <text>] [--negative <text>]
//               [--width 1024] [--height 1024] [--steps 50] [--guidance 3] [--seed 42] [--models-root <dir>]
//               [--out <png>] [--runs 2]
//
// --width and --height are snapped as the worker snaps an MES size (multiples of 16, at most 1456² in area); the JSON
// line reports the size drawn.

var args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    defer { args.removeSubrange(i...(i + 1)) }
    return args[i + 1]
}
func seconds(since start: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - start
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}

let started = ContinuousClock.now
guard let model = Model(rawValue: option("--model") ?? Model.sft.rawValue) else {
    FileHandle.standardError.write(Data("unknown --model; one of \(Model.allCases.map(\.rawValue))\n".utf8))
    exit(2)
}
guard let lane = Lane(rawValue: option("--lane") ?? Lane.bf16.rawValue) else {
    FileHandle.standardError.write(Data("unknown --lane; one of \(Lane.allCases.map(\.rawValue))\n".utf8))
    exit(2)
}
let prompt = option("--prompt") ?? "a lighthouse on a rocky coast at dusk, a sign on it that reads \"GAEA\""
let negative = option("--negative")
let size = try snapped((Int(option("--width") ?? "") ?? 1024, Int(option("--height") ?? "") ?? 1024))
let steps = Int(option("--steps") ?? "") ?? model.defaultSteps
let guidance = Double(option("--guidance") ?? "") ?? model.defaultGuidance
let seed = UInt64(option("--seed") ?? "") ?? 42
let outPath = option("--out")
let runs = Int(option("--runs") ?? "") ?? 2
let modelsRoot = URL(fileURLWithPath: option("--models-root")
                     ?? NSHomeDirectory() + "/Library/Application Support/MLXEngine/Models", isDirectory: true)

var result: [String: Any] = ["model": model.rawValue, "lane": lane.rawValue, "width": size.width, "height": size.height,
                             "steps": steps, "guidance": guidance, "seed": seed]
if let warning = size.warning { result["warning"] = warning.code }

// The budget Gaea sends a worker on a large Mac (wiresWeights off, the automatic pool cap).
let engine = MLXServeEngine(workerBudget: WorkerBudget(ceilingBytes: 64_000_000_000, wiresWeights: false))
await engine.useModelStore(ModelStore(root: modelsRoot))
let id = PackageID(model.rawValue)
let registration = model == .sft ? MossImageSFTPackage.registration : MossImageBasePackage.registration
_ = try await engine.register(registration, configuration: model.configuration(lane: lane), id: id)

var clock = ContinuousClock.now
_ = try await engine.prepare(.textToImage, package: id)
result["prepare_s"] = seconds(since: clock)

// The worker's request for its first image (MossPlugin's generate, count 1).
let request = generateRequest(prompt: prompt, negative: negative, width: size.width, height: size.height, steps: steps,
                              guidance: guidance, seed: seed)

for run in 1...max(1, runs) {
    clock = ContinuousClock.now
    let response = try await engine.run(request, package: id)
    result["run\(run)_s"] = seconds(since: clock)
    if run == 1, let image = (response as? T2IResponse)?.image, let outPath {
        try image.data.write(to: URL(fileURLWithPath: outPath))
    }
}
result["cold_s"] = seconds(since: started) - (runs > 1 ? result["run\(runs)_s"] as! Double : 0)

let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
