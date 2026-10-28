using System.Diagnostics;
using System.Text.Json;
using Wilding.Simulation;

var failures = new List<string>();
void Check(string name, Action action) {
    try { action(); Console.WriteLine($"PASS {name}"); }
    catch (Exception e) { failures.Add(name); Console.WriteLine($"FAIL {name}: {e.Message}"); }
}
void Assert(bool condition, string message) { if (!condition) throw new Exception(message); }

Check("seeded world is reproducible", () => {
    var a = new World(42, 120); var b = new World(42, 120);
    for (int i=0;i<300;i++) { a.Step(); b.Step(); }
    Assert(a.ToJson()==b.ToJson(), "Identical seeds diverged");
});
Check("grazing depletes resources and starvation removes creatures", () => {
    var w = new World(3, 0); var p = w.FindLand();
    w.AddCreature(Species.Grazer, p.X, p.Z, new Traits(.5f,.5f,.5f), 1);
    var tile=w.Tiles[w.Index(p.X,p.Z)]; tile.Food=1;
    w.Config.ResourceRegrowth=0;
    w.Step(); Assert(tile.Food<1,"Grazer did not consume local food");
    foreach (var t in w.Tiles) t.Food=0;
    for (int i=0;i<500;i++)w.Step();
    Assert(w.Creatures.Count==0,"Starved grazer survived without food");
});
Check("regeneration respects finite resource capacity", () => {
    var w=new World(5,0); foreach(var t in w.Tiles)t.Food=0;
    for(int i=0;i<100;i++)w.Step();
    Assert(w.Tiles.Any(t=>t.Food>0),"No resource regeneration");
    Assert(w.Tiles.All(t=>t.Food>=0&&t.Food<=t.Capacity),"Resource exceeded capacity");
});
Check("birth conserves energy with positive reproduction cost", () => {
    var w=new World(8,0);var p=w.FindLand();
    var parent=w.AddCreature(Species.Grazer,p.X,p.Z,new Traits(.5f,.5f,.5f),2)!;
    parent.Energy=45;parent.Age=200;
    float before=parent.Energy;var child=w.TryReproduce(parent);
    Assert(child!=null,"Mature energetic parent failed to reproduce");
    Assert(parent.Energy+child!.Energy<before,"Birth created free energy");
    Assert(child.Generation==1&&child.ParentId==parent.Id,"Ancestry not preserved");
    Assert(child.Traits.IsValid,"Mutation escaped trait bounds");
});
Check("water painting evacuates creatures and rejects invalid introductions", () => {
    var w=new World(9,20);var a=w.Creatures[0];int x=(int)a.X,z=(int)a.Z;
    w.Paint(x,z,Brush.Water,2);
    Assert(w.Creatures.All(c=>w.IsLand((int)c.X,(int)c.Z)),"Creature stranded in water");
    Assert(w.AddCreature(Species.Grazer,x,z,new Traits(.5f,.5f,.5f),0)==null,"Creature introduced into water");
});
Check("snapshot preserves exact continuation and fork independence", () => {
    var w=new World(17,100);for(int i=0;i<120;i++)w.Step();
    w.Paint(24,24,Brush.Forest,2);var copy=World.FromJson(w.ToJson());
    for(int i=0;i<200;i++){w.Step();copy.Step();}
    Assert(w.ToJson()==copy.ToJson(),"Save/load lost state or RNG");
    var fork=World.FromJson(w.ToJson());fork.Config.ResourceRegrowth=0;
    Assert(w.Config.ResourceRegrowth>0,"Fork shares mutable config");
});
Check("empty population reports no trait observation", () => {
    var w=new World(4,0);var s=w.TakeSample();
    Assert(s.Grazers.Count==0&&s.Grazers.Hearing.Median==null,"Extinction fabricated trait data");
});
Check("cap is explicit and births do not exceed it", () => {
    var w=new World(6,1);w.Config.PopulationCap=1;var p=w.Creatures[0];p.Energy=45;p.Age=200;
    Assert(w.TryReproduce(p)==null,"Population cap exceeded");
    Assert(w.CapBirthsBlocked==1,"Cap intervention not counted");
});
Check("mutation disabled yields exact inherited traits", () => {
    var w=new World(7,1);w.Config.MutationScale=0;var p=w.Creatures[0];p.Energy=45;p.Age=200;
    var child=w.TryReproduce(p)!;Assert(child.Traits==p.Traits,"Zero mutation changed genotype");
});
Check("long run has finite bounded traits and actual telemetry", () => {
    var w=new World(29,1000);for(int i=0;i<600;i++)w.Step();
    Assert(w.Creatures.All(c=>c.Traits.IsValid&&float.IsFinite(c.Energy)),"Nonfinite population state");
    Assert(w.TotalBirths>0,"No births in default world");
    Assert(w.History.Count>1&&w.History[^1].Tick>0,"No real history");
});

Check("spatial queries detect creatures crossing a bucket boundary this tick", () => {
    var w=new World(31,0);foreach(var tile in w.Tiles)tile.Food=0;
    w.AddCreature(Species.Grazer,23.99f,30.5f,new Traits(0,0,0),0);
    var hunter=w.AddCreature(Species.Hunter,28.05f,30.5f,new Traits(0,0,0),0)!;hunter.Heading=0;
    w.Tiles[w.Index(24,30)].Food=2;
    var options=new JsonSerializerOptions{IncludeFields=true};var s=JsonSerializer.Deserialize<World.Snapshot>(w.ToJson(),options)!;
    s.State=1;while(new RandomStream(s.State).Index(2)!=1)s.State++;
    w=World.FromJson(JsonSerializer.Serialize(s,options));w.Step();
    Assert(w.Creatures[0].X>24,"Probe did not cross the bucket boundary");
    Assert(w.Creatures[1].X<28.05f,"Hunter missed the now-visible grazer and wandered east");
});
Check("invalid snapshot state is rejected before it can reach the renderer", () => {
    var w=new World(22,10);var options=new JsonSerializerOptions{IncludeFields=true};
    var corruptions=new Action<World.Snapshot>[] {
        s=>s.Creatures[0].Species=(Species)2,
        s=>s.Creatures[0].Lineage=-1,
        s=>s.Tiles[0].Food=999,
        s=>s.Config.MutationScale=-1,
        s=>s.State=0,
        s=>s.History[0].Grazers.Hearing.Bins=new int[3],
        s=>s.Tiles[(int)s.Creatures[0].Z*World.Width+(int)s.Creatures[0].X].Height=0,
    };
    foreach(var corrupt in corruptions){var s=JsonSerializer.Deserialize<World.Snapshot>(w.ToJson(),options)!;corrupt(s);bool rejected=false;try{World.FromJson(JsonSerializer.Serialize(s,options));}catch(InvalidDataException){rejected=true;}Assert(rejected,"Accepted a corrupted snapshot");}
});
Check("forest painting is idempotent across repeated drag events", () => {
    var w=new World(12,0);var p=w.FindLand();w.Tiles[w.Index(p.X,p.Z)].Forest=false;
    w.Paint(p.X,p.Z,Brush.Forest,1);w.Paint(p.X,p.Z,Brush.Forest,1);
    Assert(w.Tiles[w.Index(p.X,p.Z)].Forest,"Repeated forest brush erased the forest");
});

if(failures.Count>0){Console.Error.WriteLine($"{failures.Count} checks failed");return 1;}
if(args.Contains("--benchmark")) {
    var results=new List<object>();
    foreach(int n in new[]{1000,3000}) {
        var initial=new World(42,n);string snapshot=initial.ToJson();
        var ms=new double[200];int minPopulation=int.MaxValue,maxPopulation=0;
        // Each timed step starts with exactly n creatures. Restore time is excluded.
        for(int i=0;i<ms.Length;i++){var w=World.FromJson(snapshot);long start=Stopwatch.GetTimestamp();w.Step();ms[i]=Stopwatch.GetElapsedTime(start).TotalMilliseconds;minPopulation=Math.Min(minPopulation,w.Creatures.Count);maxPopulation=Math.Max(maxPopulation,w.Creatures.Count);}
        Array.Sort(ms);results.Add(new{workload="one-step snapshot replay; restore excluded; not a sustained ecology benchmark",initialPopulation=n,minPopulationAfterStep=minPopulation,maxPopulationAfterStep=maxPopulation,measuredSteps=ms.Length,medianMs=ms[100],p95Ms=ms[190],maxMs=ms[^1]});
    }
    Console.WriteLine(JsonSerializer.Serialize(results,new JsonSerializerOptions{WriteIndented=true}));
}
Console.WriteLine("All simulation checks passed.");return 0;
