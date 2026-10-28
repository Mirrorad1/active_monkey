# Wilding

A desktop population sandbox: a stepped 3D island, pixel-art grazers and hunters,
editable habitats, and charts drawn from the running simulation.

This is the first playable game model. Its local policies and cost functions are
provided design choices. It is **not** a verified port of the Python ecology
engine, an active-inference learner, or evidence for a new scientific result.

## Play

On the development machine, tools are already prepared:

```sh
./wilding/run.sh
```

On another Apple Silicon Mac:

```sh
./wilding/setup.sh
./wilding/run.sh
```

Setup downloads official Godot .NET 4.7.2 and .NET SDK 8.0.427 archives, verifies
their hashes, and extracts them under the ignored `wilding/.tools/` directory.
There is no machine-wide installation. Downloads require internet access.

Other desktops: install Godot .NET 4.7.2 and .NET SDK 8, then set `WILDING_GODOT`
to the editor executable and optionally `WILDING_DOTNET` to `dotnet`. Adjust the
`bundled-godot` source in `NuGet.Config` to the installed editor's
`GodotSharp/Tools/nupkgs` directory, or remove that local source and use NuGet.
Only the Apple Silicon macOS build has been run here. C# web export is not part
of this version.

## Controls

- Click **Play**; select 1×, 5× or 20×. Base speed is 10 simulation ticks/second.
- Right-drag to rotate; scroll to zoom; WASD to pan. **Whole island** resets the view.
- **Watch grazers/hunters** moves the camera close enough to inspect their sprites.
- **Observe** and click a creature to read its identity, lineage, generation,
  energy and inherited hearing, speed and size.
- Select a habitat brush, then click or drag. Forest reduces food capacity and
  slows movement. Refuges prevent hunters from targeting prey inside them.
- Food refills a patch; raise land creates or raises terrain; water removes land.
  Water painting explicitly relocates affected creatures to nearest remaining
  land. If none remains, they die. This intervention is not natural selection.
- Introduce five grazers or hunters per click. They receive the displayed
  model's provided founder traits rather than inheriting from nearby creatures.
- Natural, food availability and lineage overlays expose different world state.
  Food colors update periodically without rebuilding the terrain mesh.
- Save and load preserve the exact simulation state and random stream. Load
  pauses. **New island** replaces the current world; save first if needed.
- Fork saves the original state to a timestamped file and continues an independent
  copy. To compare branches, preserve both files; simultaneous side-by-side
  playback is not implemented yet.

Save files and CSV exports appear under Godot's user-data directory, usually
`~/Library/Application Support/Godot/app_userdata/Wilding/` on macOS. The status
bar reports the exact location. Filenames: `alder-island.json`,
`fork-<tick>-<timestamp>.json`, and `traits.csv`.

## What is simulated

- Finite food regenerates locally to a capacity. Grazers eat local food and
  use a provided local foraging rule; hunters track nearby prey and attack.
- Hearing changes detection range and costs energy. Speed changes movement
  distance and movement cost. Body size affects intake, upkeep and lifespan.
- Creatures reproduce asexually after maturity when their own energy exceeds
  a threshold. A birth transfers 8 energy to the offspring and subtracts 11
  from its parent. Newborn traits mutate within [0,1].
- Predation, energy exhaustion and age remove creatures. Mutation and genetic
  drift can change populations without an adaptive explanation.
- Spatial buckets accelerate local searches. Processing order is shuffled each
  tick using the saved random stream. Rendering does not consume that stream.
- The initial population is 1,000 (940 grazers, 60 hunters). Boom, collapse and
  extinction are possible; coexistence is not imposed.
- A hard cap of 6,000 total creatures is an engineering limit. Blocked births
  are counted and shown. Cap saturation must not be interpreted as an ecological
  equilibrium.

There is no within-life learning, social communication, speciation criterion,
sexual recombination or full mass/energy conservation model. The three founder
lineages are assigned labels, not species. Sprite shape indicates ecological
role, and juvenile size is visual age scaling.

## Reading the charts

Choose a population, inherited trait, and **living creatures** or **newborns per
sample interval**. The green line is the median; the shaded band covers the
10th–90th percentiles of actual individuals, not confidence in an estimate.
The histogram compares the selected cohort with the original founder genotypes.
No observation is drawn for an empty cohort.

Samples are recorded every 20 ticks; the last 256 are retained, plus the founder
summary. The horizontal axis is **simulation ticks**. Generations overlap, so the
maximum lineage generation in the clock is not used as a population-wide clock.
The scrubber selects historical samples; **Live** follows the latest one.

Newborns are all births in the preceding sampling interval, including newborns
that died before its end. Comparing newborn and living distributions helps
distinguish inheritance from survivor composition. Neither chart alone establishes
a causal selection effect. All charts are sampled telemetry, not mockup data.

## Verification

```sh
./wilding/run.sh check
./wilding/run.sh benchmark
./wilding/run.sh headless
./wilding/run.sh smoke
./wilding/run.sh editor
```

The dependency-free checks cover deterministic replay, resource bounds,
starvation, energy transfer at birth, ancestry/mutation, terrain evacuation,
snapshot continuation and isolation, extinction, cap accounting, spatial
bucket crossings, and invalid snapshot rejection. Graphical smoke verifies
the running clock, terrain picking, zoom input, actual save/load, chart cohort
selection and live food-overlay updates; it writes `verification/smoke.png`.

`verification/core-checks.txt` contains the controlled simulation benchmark.
Every timed step starts from an identical snapshot containing exactly 1,000 or
3,000 creatures. Snapshot restoration is excluded from timing. This measures
one-step update workload, **not sustained ecology throughput or rendered FPS**.
Long-run population loss would otherwise make a nominal 1,000-creature benchmark
misleading. Timings are machine-specific and include normal measurement noise.

## Layout

- `Simulation/`: engine-independent C# state, rules, RNG, validation and telemetry.
- `Game/`: Godot world rendering, batched sprites, camera, controls and charts.
- `Checks/`: console checks and controlled benchmark; no test-framework dependency.
- `verification/`: retained engineering verification and screenshot.

Terrain uses chunk meshes; trees and creatures use GPU instancing. The model
uses fixed ticks with a bounded per-frame catch-up budget. Camera position
does not change the ecology rules. Performance optimization should preserve
these boundaries and be checked against seeded snapshots.
