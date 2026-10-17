import Cocoa
import FlutterMacOS

class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    self.contentViewController = flutterViewController

    // The explorer shows a result list plus a 400 pt detail rail.
    self.minSize = NSSize(width: 1100, height: 700)
    let target = NSSize(width: 1500, height: 950)
    self.setContentSize(target)
    self.center()
    self.title = "Chroma Retrieval Explorer"

    RegisterGeneratedPlugins(registry: flutterViewController)

    super.awakeFromNib()
  }
}
