namespace Wilding.Simulation;

// State belongs to the snapshot; rendering never consumes this stream.
public sealed class RandomStream {
    public ulong State;
    public RandomStream(ulong state) { State=state==0?0x9e3779b97f4a7c15UL:state; }
    public uint NextUInt() { ulong x=State;x^=x>>12;x^=x<<25;x^=x>>27;State=x;return (uint)((x*2685821657736338717UL)>>32); }
    public float Next()=> (NextUInt()>>8)*(1f/16777216f);
    public int Index(int count)=>Math.Min(count-1,(int)(Next()*count));
    public float Noise()=> (Next()+Next()+Next()-1.5f)*1.41421356f;
}
