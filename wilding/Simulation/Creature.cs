namespace Wilding.Simulation;
public enum Species { Grazer, Hunter }
public enum Brush { Observe, Forest, Water, Refuge, Food, Raise }
public readonly record struct Traits(float Hearing,float Speed,float Size) {
    public bool IsValid=>float.IsFinite(Hearing)&&float.IsFinite(Speed)&&float.IsFinite(Size)&&Hearing is >=0 and <=1&&Speed is >=0 and <=1&&Size is >=0 and <=1;
    public float Value(int trait)=>trait==0?Hearing:trait==1?Speed:Size;
    public Traits Mutate(RandomStream rng,float scale)=>new(Math.Clamp(Hearing+rng.Noise()*scale,0,1),Math.Clamp(Speed+rng.Noise()*scale,0,1),Math.Clamp(Size+rng.Noise()*scale,0,1));
}
public sealed class Tile { public int Height;public bool Forest,Refuge;public float Food,Capacity; }
public sealed class Creature {
    public long Id,ParentId;public Species Species;public int Generation,Age,Lineage;public float X,Z,Energy,Heading;public bool Alive=true;public Traits Traits;
}
public sealed class Config { public float ResourceRegrowth=.025f,MutationScale=.035f;public int PopulationCap=6000; }
public sealed record WorldEvent(int Tick,string Text);
