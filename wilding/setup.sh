#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  printf '%s\n' 'Automatic setup currently supports Apple Silicon macOS. Install Godot .NET 4.7.2 and .NET SDK 8 manually on other platforms; see README.md.' >&2
  exit 1
fi
mkdir -p .tools/dotnet
if [ ! -x .tools/dotnet/dotnet ]; then
  curl -fL --retry 2 https://builds.dotnet.microsoft.com/dotnet/Sdk/8.0.427/dotnet-sdk-8.0.427-osx-arm64.tar.gz -o .tools/dotnet-sdk.tar.gz
  printf '%s\n' '79786b93f249d233b15f1d6338d4798bd71c8143f267c5e84de76c9eb8a1054920e56ea5478f7bddfb74b951cc58aaca11fbd780cf134350b7cb7cdcb42a8c8c  .tools/dotnet-sdk.tar.gz' | shasum -a 512 -c -
  tar -xzf .tools/dotnet-sdk.tar.gz -C .tools/dotnet
  rm .tools/dotnet-sdk.tar.gz
fi
if [ ! -x .tools/Godot_mono.app/Contents/MacOS/Godot ]; then
  curl -fL --retry 2 https://github.com/godotengine/godot/releases/download/4.7.2-stable/Godot_v4.7.2-stable_mono_macos.universal.zip -o .tools/godot.zip
  printf '%s\n' '8af3977b60d2c59802f7c8ff1914b3ca02a5e294f7381fc1104ee777e33cbbd8  .tools/godot.zip' | shasum -a 256 -c -
  ditto -xk .tools/godot.zip .tools
  rm .tools/godot.zip
fi
printf '%s\n' 'Build tools ready. Run ./wilding/run.sh from the repository root.'
