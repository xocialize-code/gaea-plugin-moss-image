import GaeaPluginKit
import MossPlugin

// The plugin's worker: MossPlugin's definition over GPP (GaeaPluginKit). `--manifest <path>` writes manifest.json.
await GaeaPluginMain.run(mossPlugin)
