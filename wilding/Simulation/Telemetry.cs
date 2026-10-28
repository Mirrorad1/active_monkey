namespace Wilding.Simulation;
public sealed class TraitStats {
    public float? Median,Low,High;public int[] Bins=new int[10];
    public static TraitStats Measure(IEnumerable<float> values) {
        var sorted=values.Order().ToArray();var result=new TraitStats();if(sorted.Length==0)return result;
        float Q(float p) { float i=(sorted.Length-1)*p;int lo=(int)i;return sorted[lo]+(sorted[Math.Min(lo+1,sorted.Length-1)]-sorted[lo])*(i-lo); }
        result.Median=Q(.5f);result.Low=Q(.1f);result.High=Q(.9f);
        foreach(float v in sorted)result.Bins[Math.Min(9,(int)(v*10))]++;return result;
    }
}
public sealed class CohortStats {
    public int Count;public TraitStats Hearing=new(),Speed=new(),Size=new();
    public TraitStats Trait(int id)=>id==0?Hearing:id==1?Speed:Size;
    public static CohortStats Measure(IEnumerable<Traits> source) {
        var a=source.ToArray();return new(){Count=a.Length,Hearing=TraitStats.Measure(a.Select(t=>t.Hearing)),Speed=TraitStats.Measure(a.Select(t=>t.Speed)),Size=TraitStats.Measure(a.Select(t=>t.Size))};
    }
}
public sealed class Sample {
    public int Tick,MaxGeneration;public long Births,Deaths,CapBirthsBlocked;
    public CohortStats Grazers=new(),Hunters=new(),NewGrazers=new(),NewHunters=new();
    public CohortStats Cohort(Species species,bool newborn)=>species==Species.Grazer?(newborn?NewGrazers:Grazers):(newborn?NewHunters:Hunters);
}
