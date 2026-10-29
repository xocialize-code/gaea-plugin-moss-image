// swift-tools-version: 6.2
import PackageDescription

// gaea-plugin-moss-image: OpenMOSS's MOSS-Image (SFT and Base, Apache-2.0) as a Gaea research plugin. One worker
// executable built on GaeaPluginKit over moss-image-swift, packaged as a signed .gaeaplugin by scripts/build-plugin.sh.
//
// moss-inproc is not part of the plugin: it runs the same packages in-process through MLXServeEngine, with the same
// dependency resolution, as the reference a run through Gaea is compared against.
//
// GAEA_PLUGIN_KIT_PATH points the kit at a local checkout while you work on both.
let kitPath = Context.environment["GAEA_PLUGIN_KIT_PATH"]

let package = Package(
    name: "gaea-plugin-moss-image",
    platforms: [.macOS(.v26)], // mlx-engine-swift's MLXToolKit requires macOS 26
    dependencies: [
        kitPath.map { .package(path: $0) }
            ?? .package(url: "https://github.com/xocialize/gaea-plugin-kit.git", from: "0.3.0"),
        .package(url: "https://github.com/xocialize/moss-image-swift.git", exact: "0.3.0"),
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", from: "0.72.0"),
    ],
    targets: [
        .target(
            name: "MossPlugin",
            dependencies: [
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
                .product(name: "MLXMossImage", package: "moss-image-swift"),
                // MossImageTextEncoderQuant, the encoder tier a lane sets, is the core's type.
                .product(name: "MossImage", package: "moss-image-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ]
        ),
        .executableTarget(
            name: "gaea-moss-image-worker",
            dependencies: [
                "MossPlugin",
                .product(name: "GaeaPluginKit", package: "gaea-plugin-kit"),
            ]
        ),
        .executableTarget(
            name: "moss-inproc",
            dependencies: [
                "MossPlugin",
                .product(name: "MLXMossImage", package: "moss-image-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
