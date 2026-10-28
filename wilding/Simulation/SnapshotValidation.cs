namespace Wilding.Simulation;

public static class SnapshotValidation {
    public static void Validate(World.Snapshot s) {
        void Require(bool condition){if(!condition)throw new InvalidDataException("Invalid or unsupported world snapshot");}
        Require(s.Version==1&&s.State!=0&&s.Tick>=0&&s.Revision>=0&&s.MaxGeneration>=0&&s.NextId>0&&s.TotalBirths>=0&&s.TotalDeaths>=0&&s.CapBirthsBlocked>=0);
        Require(s.Config!=null&&s.Tiles!=null&&s.Creatures!=null&&s.History!=null&&s.Events!=null&&s.PendingGrazers!=null&&s.PendingHunters!=null&&s.Founders!=null);
        var config=s.Config!;
        Require(float.IsFinite(config.ResourceRegrowth)&&config.ResourceRegrowth is >=0 and <=10&&float.IsFinite(config.MutationScale)&&config.MutationScale is >=0 and <=1&&config.PopulationCap is >=1 and <=8192);
        Require(s.Tiles!.Length==World.Width*World.Width&&s.Creatures!.Count<=config.PopulationCap&&s.History!.Count is >=1 and <=256);
        foreach(var t in s.Tiles){Require(t!=null);Require(t!.Height is >=0 and <=7&&float.IsFinite(t.Food)&&float.IsFinite(t.Capacity)&&t.Capacity>=0&&t.Food>=0&&t.Food<=t.Capacity+.00001f);if(t.Height==0)Require(t.Capacity==0&&t.Food==0&&!t.Forest&&!t.Refuge);else Require(t.Capacity>0);}
        var ids=new HashSet<long>();
        foreach(var c in s.Creatures!) {
            Require(c!=null);Require(Enum.IsDefined(c!.Species)&&c.Lineage>=0&&c.Generation>=0&&c.Age>=0&&c.ParentId>=0&&c.Id>0&&c.Id<s.NextId&&ids.Add(c.Id)&&c.Alive&&c.Traits.IsValid&&float.IsFinite(c.X)&&float.IsFinite(c.Z)&&float.IsFinite(c.Energy)&&c.Energy>0&&float.IsFinite(c.Heading)&&c.X>=0&&c.Z>=0&&c.X<World.Width&&c.Z<World.Width);
            Require(s.Tiles[(int)c.Z*World.Width+(int)c.X].Height>0);
        }
        void Stats(TraitStats? t,int count) {
            Require(t!=null&&t.Bins!=null&&t.Bins.Length==10);var bins=t!.Bins!;Require(bins.All(b=>b>=0)&&bins.Sum(b=>(long)b)==count);
            if(count==0)Require(t.Median==null&&t.Low==null&&t.High==null);
            else Require(t.Low is float lo&&t.Median is float med&&t.High is float hi&&float.IsFinite(lo)&&float.IsFinite(med)&&float.IsFinite(hi)&&lo>=0&&lo<=med&&med<=hi&&hi<=1);
        }
        void Cohort(CohortStats? c){Require(c!=null&&c.Count>=0);Stats(c!.Hearing,c.Count);Stats(c.Speed,c.Count);Stats(c.Size,c.Count);}
        void Sample(Sample? p){Require(p!=null&&p.Tick>=0&&p.Tick<=s.Tick&&p.MaxGeneration>=0);Cohort(p!.Grazers);Cohort(p.Hunters);Cohort(p.NewGrazers);Cohort(p.NewHunters);}
        Sample(s.Founders);int previous=-1;foreach(var sample in s.History!){Sample(sample);Require(sample.Tick>=previous);previous=sample.Tick;}
        Require(s.PendingGrazers!.All(t=>t.IsValid)&&s.PendingHunters!.All(t=>t.IsValid));
        foreach(var e in s.Events!)Require(e!=null&&e.Tick>=0&&e.Tick<=s.Tick&&e.Text!=null);
    }
}
