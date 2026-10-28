using System.Text.Json;
namespace Wilding.Simulation;

public sealed class World {
    public const int Width=64;
    public Config Config=new();public List<Creature> Creatures=new();public Tile[] Tiles=new Tile[Width*Width];
    public List<Sample> History=new();public List<WorldEvent> Events=new();
    public int Seed,Tick,Revision,MaxGeneration;public long TotalBirths,TotalDeaths,CapBirthsBlocked,NextId=1;
    public List<Traits> PendingGrazers=new(),PendingHunters=new();public Sample? Founders;
    readonly RandomStream rng;
    const int BucketSize=4,Buckets=Width/BucketSize;
    readonly List<int>[] preyBuckets=new List<int>[Buckets*Buckets],hunterBuckets=new List<int>[Buckets*Buckets];
    int[] order=Array.Empty<int>();
    public World(int seed,int population) {
        Seed=seed;rng=new RandomStream((ulong)(uint)seed+1);
        for(int i=0;i<preyBuckets.Length;i++){preyBuckets[i]=new();hunterBuckets[i]=new();}
        for(int z=0;z<Width;z++)for(int x=0;x<Width;x++) {
            float dx=(x-31.5f)/28,dz=(z-31.5f)/25;
            double edge=1-dx*dx-dz*dz+.06*Math.Sin(x*.43)+.05*Math.Cos(z*.48);
            bool river=Math.Abs(x-34-Math.Sin(z*.18)*4)<1.1&&z>14&&z<55;
            int height=edge<.03||river?0:1+(int)(edge*.8+4*Math.Exp(-((x-22)*(x-22)/110.0+(z-23)*(z-23)/90.0)));
            bool forest=height>1&&height<5&&rng.Next()<.28f;float cap=height==0?0:forest?1.8f:2.6f;
            Tiles[Index(x,z)]=new(){Height=height,Forest=forest,Capacity=cap,Food=cap*(.6f+.4f*rng.Next())};
        }
        for(int i=0;i<population;i++) {
            var p=FindLand();var traits=new Traits(.25f+.45f*rng.Next(),.25f+.45f*rng.Next(),.25f+.45f*rng.Next());
            AddCreature(i>=population*.94f?Species.Hunter:Species.Grazer,p.X+.5f,p.Z+.5f,traits,i%3);
        }
        Founders=TakeSample();History.Add(Founders);Events.Add(new(0,$"Island seeded with {Creatures.Count} creatures"));
    }
    public int Index(int x,int z)=>z*Width+x;
    public bool IsLand(int x,int z)=>x>=0&&z>=0&&x<Width&&z<Width&&Tiles[Index(x,z)].Height>0;
    public (int X,int Z) FindLand() {
        for(int i=0;i<4096;i++){int x=rng.Index(Width),z=rng.Index(Width);if(IsLand(x,z))return(x,z);}
        for(int i=0;i<Tiles.Length;i++)if(Tiles[i].Height>0)return(i%Width,i/Width);
        throw new InvalidOperationException("The world has no land");
    }
    public Creature? AddCreature(Species species,float x,float z,Traits traits,int lineage) {
        if(!float.IsFinite(x)||!float.IsFinite(z)||!traits.IsValid||!IsLand((int)x,(int)z)||x<0||z<0||Creatures.Count>=Config.PopulationCap)return null;
        var c=new Creature{Id=NextId++,Species=species,X=x,Z=z,Traits=traits,Lineage=lineage,Energy=12+rng.Next()*5,Heading=rng.Next()*MathF.Tau};Creatures.Add(c);return c;
    }
    int Bucket(float x,float z)=>Math.Clamp((int)z/BucketSize,0,Buckets-1)*Buckets+Math.Clamp((int)x/BucketSize,0,Buckets-1);
    Creature? Nearest(Creature c,float radius,List<int>[] buckets) {
        Creature? best=null;float distance=radius*radius;
        // Buckets describe the beginning of this tick; an earlier actor can have moved
        // at most (.10 + 1*.18)*1.12 = .3136 units since insertion.
        float scan=radius+.314f;
        int loX=Math.Max(0,(int)(c.X-scan)/BucketSize),hiX=Math.Min(Buckets-1,(int)(c.X+scan)/BucketSize);
        int loZ=Math.Max(0,(int)(c.Z-scan)/BucketSize),hiZ=Math.Min(Buckets-1,(int)(c.Z+scan)/BucketSize);
        for(int z=loZ;z<=hiZ;z++)for(int x=loX;x<=hiX;x++)foreach(int index in buckets[z*Buckets+x]) {
            var other=Creatures[index];if(!other.Alive)continue;
            if(c.Species==Species.Hunter&&Tiles[Index((int)other.X,(int)other.Z)].Refuge)continue;
            float d=(other.X-c.X)*(other.X-c.X)+(other.Z-c.Z)*(other.Z-c.Z);if(d<distance){distance=d;best=other;}
        }
        return best;
    }
    public void Step() {
        Tick++;foreach(var t in Tiles)if(t.Height>0)t.Food=Math.Min(t.Capacity,t.Food+Config.ResourceRegrowth*(t.Forest?.7f:1));
        foreach(var b in preyBuckets)b.Clear();foreach(var b in hunterBuckets)b.Clear();
        int count=Creatures.Count;if(order.Length<count)order=new int[Math.Max(count*2,128)];
        for(int i=0;i<count;i++){var c=Creatures[i];(c.Species==Species.Grazer?preyBuckets:hunterBuckets)[Bucket(c.X,c.Z)].Add(i);order[i]=i;}
        for(int i=count-1;i>0;i--){int j=rng.Index(i+1);(order[i],order[j])=(order[j],order[i]);}
        for(int k=0;k<count;k++) {
            var c=Creatures[order[k]];if(!c.Alive)continue;c.Age++;float dx=0,dz=0;var tile=Tiles[Index((int)c.X,(int)c.Z)];
            if(c.Species==Species.Grazer) {
                var danger=Nearest(c,1.8f+c.Traits.Hearing*5,hunterBuckets);
                if(danger!=null&&!tile.Refuge){dx=c.X-danger.X;dz=c.Z-danger.Z;}
                else if(tile.Food<.25f||rng.Next()<.13f) {
                    float best=tile.Food*.85f;
                    for(int iz=-1;iz<=1;iz++)for(int ix=-1;ix<=1;ix++) {
                        int x=(int)c.X+ix,z=(int)c.Z+iz;if(!IsLand(x,z))continue;
                        float score=Tiles[Index(x,z)].Food/(1+.12f*(Math.Abs(ix)+Math.Abs(iz)))+rng.Next()*.08f;
                        if(score>best){best=score;dx=x+.5f-c.X;dz=z+.5f-c.Z;}
                    }
                }
                float intake=Math.Min(tile.Food,.17f+c.Traits.Size*.1f);tile.Food-=intake;c.Energy=Math.Min(48,c.Energy+intake);
            } else {
                var prey=Nearest(c,4+c.Traits.Hearing*6,preyBuckets);
                if(prey!=null) {
                    dx=prey.X-c.X;dz=prey.Z-c.Z;
                    if(dx*dx+dz*dz<.75f*.75f) {
                        c.Energy-=.035f;
                        if(rng.Next()<.17f+c.Traits.Speed*.23f-c.Traits.Size*.03f){prey.Alive=false;TotalDeaths++;c.Energy=Math.Min(48,c.Energy+7+prey.Traits.Size*2);}
                    }
                }else{c.Heading+=(rng.Next()-.5f)*.6f;dx=MathF.Cos(c.Heading);dz=MathF.Sin(c.Heading);}
            }
            float length=MathF.Sqrt(dx*dx+dz*dz);bool moving=length>.001f;
            if(moving) {
                float step=(.10f+c.Traits.Speed*.18f)*(c.Species==Species.Hunter?1.12f:1)*(tile.Forest?.8f:1);
                float nx=c.X+dx/length*step,nz=c.Z+dz/length*step;
                if(IsLand((int)nx,(int)nz)&&nx>=0&&nz>=0){c.X=nx;c.Z=nz;c.Heading=MathF.Atan2(dz,dx);}else c.Heading+=1.5f;
            }
            // Explicit provided costs, not a hidden global fitness score.
            c.Energy-=(c.Species==Species.Hunter?.043f:.019f)+c.Traits.Hearing*.014f+c.Traits.Size*c.Traits.Size*.015f+(moving?c.Traits.Speed*c.Traits.Speed*.018f:0);
            if(c.Energy<=0||c.Age>3000+(int)(1800*(1-c.Traits.Size))){c.Alive=false;TotalDeaths++;continue;}
            if(c.Energy>26&&c.Age>90)TryReproduce(c);
        }
        Creatures.RemoveAll(c=>!c.Alive);
        if(Tick%20==0){History.Add(TakeSample());if(History.Count>256)History.RemoveAt(0);PendingGrazers.Clear();PendingHunters.Clear();}
    }
    public Creature? TryReproduce(Creature parent) {
        if(!parent.Alive||parent.Energy<26||parent.Age<90)return null;
        if(Creatures.Count>=Config.PopulationCap){CapBirthsBlocked++;return null;}
        var child=AddCreature(parent.Species,parent.X,parent.Z,parent.Traits.Mutate(rng,Config.MutationScale),parent.Lineage);if(child==null)return null;
        parent.Energy-=11;child.Energy=8;child.Generation=parent.Generation+1;child.ParentId=parent.Id;
        MaxGeneration=Math.Max(MaxGeneration,child.Generation);TotalBirths++;
        (child.Species==Species.Grazer?PendingGrazers:PendingHunters).Add(child.Traits);return child;
    }
    public Sample TakeSample()=>new(){Tick=Tick,MaxGeneration=MaxGeneration,Births=TotalBirths,Deaths=TotalDeaths,CapBirthsBlocked=CapBirthsBlocked,
        Grazers=CohortStats.Measure(Creatures.Where(c=>c.Alive&&c.Species==Species.Grazer).Select(c=>c.Traits)),Hunters=CohortStats.Measure(Creatures.Where(c=>c.Alive&&c.Species==Species.Hunter).Select(c=>c.Traits)),
        NewGrazers=CohortStats.Measure(PendingGrazers),NewHunters=CohortStats.Measure(PendingHunters)};
    public void Paint(int x,int z,Brush brush,int radius) {
        if(brush==Brush.Observe)return;
        for(int iz=Math.Max(0,z-radius);iz<=Math.Min(Width-1,z+radius);iz++)for(int ix=Math.Max(0,x-radius);ix<=Math.Min(Width-1,x+radius);ix++) {
            if((ix-x)*(ix-x)+(iz-z)*(iz-z)>radius*radius)continue;var t=Tiles[Index(ix,iz)];
            if(brush==Brush.Water){t.Height=0;t.Food=t.Capacity=0;t.Forest=t.Refuge=false;}
            else if(brush==Brush.Raise){t.Height=Math.Min(7,t.Height+1);t.Capacity=t.Forest?1.8f:2.6f;}
            else if(t.Height>0){if(brush==Brush.Forest){t.Forest=true;t.Refuge=false;t.Capacity=1.8f;t.Food=Math.Min(t.Food,t.Capacity);}if(brush==Brush.Refuge){t.Refuge=true;t.Forest=false;t.Capacity=2.6f;}if(brush==Brush.Food)t.Food=t.Capacity;}
        }
        foreach(var c in Creatures)if(!IsLand((int)c.X,(int)c.Z)) {
            int best=-1;float distance=float.PositiveInfinity;
            for(int i=0;i<Tiles.Length;i++)if(Tiles[i].Height>0){float q=(i%Width+.5f-c.X)*(i%Width+.5f-c.X)+(i/Width+.5f-c.Z)*(i/Width+.5f-c.Z);if(q<distance){distance=q;best=i;}}
            if(best<0){c.Alive=false;TotalDeaths++;}else{c.X=best%Width+.5f;c.Z=best/Width+.5f;}
        }
        Creatures.RemoveAll(c=>!c.Alive);Revision++;Events.Add(new(Tick,$"{brush} brush at {x}, {z}"));if(Events.Count>64)Events.RemoveAt(0);
    }
    static readonly JsonSerializerOptions JsonOptions=new(){IncludeFields=true};
    public string ToJson()=>JsonSerializer.Serialize(new Snapshot{Version=1,Seed=Seed,Tick=Tick,Revision=Revision,MaxGeneration=MaxGeneration,Config=Config,Tiles=Tiles,Creatures=Creatures,History=History,Founders=Founders,Events=Events,State=rng.State,NextId=NextId,TotalBirths=TotalBirths,TotalDeaths=TotalDeaths,CapBirthsBlocked=CapBirthsBlocked,PendingGrazers=PendingGrazers,PendingHunters=PendingHunters},JsonOptions);
    public static World FromJson(string json) {
        var s=JsonSerializer.Deserialize<Snapshot>(json,JsonOptions)??throw new InvalidDataException("Empty snapshot");
        SnapshotValidation.Validate(s);
        var w=new World(s.Seed,0){Tick=s.Tick,Revision=s.Revision,MaxGeneration=s.MaxGeneration,Config=s.Config,Tiles=s.Tiles,Creatures=s.Creatures,History=s.History,Founders=s.Founders,Events=s.Events,NextId=s.NextId,TotalBirths=s.TotalBirths,TotalDeaths=s.TotalDeaths,CapBirthsBlocked=s.CapBirthsBlocked,PendingGrazers=s.PendingGrazers,PendingHunters=s.PendingHunters};w.rng.State=s.State;return w;
    }
    public sealed class Snapshot {
        public int Version,Seed,Tick,Revision,MaxGeneration;public Config Config=new();public Tile[] Tiles=Array.Empty<Tile>();public List<Creature> Creatures=new();public List<Sample> History=new();public Sample? Founders;public List<WorldEvent> Events=new();
        public ulong State;public long NextId,TotalBirths,TotalDeaths,CapBirthsBlocked;public List<Traits> PendingGrazers=new(),PendingHunters=new();
    }
}
