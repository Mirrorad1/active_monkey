#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .tools/dotnet/dotnet ]; then
  export DOTNET_ROOT="$PWD/.tools/dotnet"
  export PATH="$DOTNET_ROOT:$PATH"
fi
export DOTNET_CLI_HOME="${DOTNET_CLI_HOME:-$PWD/.tools/dotnet-home}"
export NUGET_PACKAGES="${NUGET_PACKAGES:-$PWD/.tools/nuget}"
export DOTNET_NOLOGO=1
dotnet_bin="${WILDING_DOTNET:-dotnet}"
godot_bin="${WILDING_GODOT:-$PWD/.tools/Godot_mono.app/Contents/MacOS/Godot}"
if ! command -v "$dotnet_bin" >/dev/null || [ ! -x "$godot_bin" ]; then
  printf '%s\n' 'Run ./wilding/setup.sh first, or set WILDING_DOTNET and WILDING_GODOT to your installed tools.' >&2
  exit 1
fi
case "${1:-play}" in
  check) exec "$dotnet_bin" run --project Checks/Checks.csproj -c Release -p:UseSharedCompilation=false ;;
  benchmark) exec "$dotnet_bin" run --project Checks/Checks.csproj -c Release -p:UseSharedCompilation=false -- --benchmark ;;
esac
"$dotnet_bin" build Wilding.csproj -p:UseSharedCompilation=false --ignore-failed-sources
case "${1:-play}" in
  smoke) exec "$godot_bin" --path "$PWD" -- --smoke ;;
  headless) exec "$godot_bin" --headless --path "$PWD" -- --smoke ;;
  editor) exec "$godot_bin" --editor --path "$PWD" ;;
  play) exec "$godot_bin" --path "$PWD" ;;
  *) printf '%s\n' 'Usage: run.sh [play|editor|check|benchmark|smoke|headless]' >&2; exit 1 ;;
esac
