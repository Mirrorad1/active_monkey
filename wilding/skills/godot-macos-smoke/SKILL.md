---
name: godot-macos-smoke
description: Verify native Godot C# game changes on macOS when headless checks pass but a graphical smoke test hangs while awaiting a redraw, or desktop rendering fails inside a sandbox. Applies to Godot engine checks, not browser mockups.
---

# Native Godot smoke checks on macOS

Use the game's own deterministic smoke mode before general native UI automation.
For Wilding, `./wilding/run.sh headless` checks controller behavior and
`./wilding/run.sh smoke` additionally captures the actual rendered window.
Treat exit code plus `WILDING_SMOKE_OK` as the completion evidence; a created
window or screenshot alone does not prove that the test finished.

## Unfocused-window redraw wait

In this environment, awaiting `RenderingServer.FramePostDraw` inside the
smoke completion routine could wait indefinitely when the native test window
was unfocused. The engine initialized and wrote a screenshot during other runs,
but never printed the completion marker in the failing run.

After the scene has rendered for several frames, the prior frame is already
available through `GetViewport().GetTexture().GetImage()`. Capturing that frame
synchronously removed the redraw dependency; the same graphical smoke then
completed normally. This capture is one frame old: do not describe immediate
UI changes performed after that frame as present in the screenshot. If exact
post-action pixels matter, advance an explicit test stage through additional
frames before capture rather than waiting on an event that may be suppressed.

Keep a real clock-advancement assertion: a smoke run can otherwise pass with
static graphics while an accumulator clamp prevents every simulation tick.
Also retain checks on actual terrain picking, save/load continuation and chart
cohort selection; static screenshots do not exercise those paths.

## Desktop permission boundary

An offscreen/headless model check does not establish access to macOS graphics.
Godot may need its normal application-data directory and desktop connection.
MuJoCo's `invalid CoreGraphics connection` also reproduced inside the managed
sandbox and disappeared when the same focused rendering tests ran with desktop
access. Request a narrowly scoped execution escalation through the command tool
when needed; do not weaken OS security settings or infer a renderer defect from
this permission failure.

For this project, the launcher finds verified local Godot/.NET tools in
`wilding/.tools/`. Bundled Godot NuGet packages are under
`GodotSharp/Tools/nupkgs`. A package-audit network warning is separate from a C#
compile error; rerun ordinary restore with authorized network access rather
than disabling auditing.

## References

- [Godot C# prerequisites](https://docs.godotengine.org/en/stable/tutorials/scripting/c_sharp/c_sharp_basics.html)
- [Headless execution](https://docs.godotengine.org/en/stable/tutorials/export/exporting_for_dedicated_servers.html)
- Verified implementation: `wilding/Game/Sandbox.cs`, `CompleteSmoke`.
