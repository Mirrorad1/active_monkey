using Godot;
using Wilding.Simulation;
namespace Wilding.Game;

public partial class ChartView:Control {
    public World? World;
    public Species Species;
    public int Trait,Selected=-1;
    public bool Newborn,Histogram;
    public Action<int>? SelectedChanged;
    readonly Color ink=new("29484a"),muted=new("617977"),grid=new("d3dfd7"),line=new("427e71"),band=new(.65f,.79f,.71f,.55f);
    readonly string[] names={"Hearing sensitivity","Movement speed","Body size"};
    int hover=-1;
    public override void _Ready(){CustomMinimumSize=new Vector2(300,200);SizeFlagsHorizontal=SizeFlags.ExpandFill;MouseFilter=MouseFilterEnum.Stop;MouseExited+=()=>{hover=-1;QueueRedraw();};}
    void Text(Vector2 p,string text,int size=12,Color? color=null)=>DrawString(ThemeDB.FallbackFont,p,text,HorizontalAlignment.Left,-1,size,color??ink);
    public override void _Draw() {
        if(World==null||World.History.Count==0)return;
        var history=World.History;int picked=Selected<0?history.Count-1:Math.Clamp(Selected,0,history.Count-1);
        var cohort=history[picked].Cohort(Species,Newborn);var current=cohort.Trait(Trait);
        Text(new(0,16),Histogram?"How individuals differ":"How the trait changes",14);
        Text(new(0,34),Histogram?$"Tick {history[picked].Tick} • {cohort.Count} {(Newborn?"newborns":"living creatures")}":"Median and middle 80% • inherited traits",11,muted);
        float left=40,right=Size.X-12,top=57,bottom=Size.Y-33,pw=right-left,ph=bottom-top;
        if(pw<80||ph<30)return;
        if(Histogram) {
            var founder=World.Founders?.Cohort(Species,false)??new CohortStats();var baseStats=founder.Trait(Trait);
            float peak=1;
            for(int i=0;i<10;i++)peak=Math.Max(peak,Math.Max(cohort.Count>0?current.Bins[i]*100f/cohort.Count:0,founder.Count>0?baseStats.Bins[i]*100f/founder.Count:0));
            float ymax=Math.Max(20,MathF.Ceiling(peak/10)*10);
            foreach(float tick in new[]{0,ymax/2,ymax}){float y=bottom-tick/ymax*ph;DrawLine(new(left,y),new(right,y),grid);Text(new(4,y+4),$"{tick:0}%",11,muted);}
            for(int i=0;i<10;i++) {
                float w=pw/10,x=left+i*w;
                if(founder.Count>0){float h=baseStats.Bins[i]*100f/founder.Count/ymax*ph;DrawRect(new Rect2(x+1,bottom-h,w-2,h),new("c8d4cd"));}
                if(cohort.Count>0){float h=current.Bins[i]*100f/cohort.Count/ymax*ph;DrawRect(new Rect2(x+w*.25f,bottom-h,w*.5f,h),new("739a88"));}
            }
            foreach(float v in new[]{0,.5f,1})Text(new(left+v*pw-5,bottom+16),$"{v:0.0}",11,muted);
            Text(new(left,Size.Y-3),$"{names[Trait]} (0–1)",11,muted);
            if(cohort.Count==0)Text(new(left+12,top+ph/2),Newborn?"No births in this sample interval":"No living creatures",13);
        }else{
            foreach(float tick in new[]{0,.5f,1}){float y=bottom-tick*ph;DrawLine(new(left,y),new(right,y),grid);Text(new(4,y+4),$"{tick:0.0}",11,muted);}
            int minTick=history[0].Tick,maxTick=Math.Max(minTick+1,history[^1].Tick);
            float X(int tick)=>left+(tick-minTick)/(float)(maxTick-minTick)*pw;
            for(int i=0;i<4;i++){int tick=minTick+(maxTick-minTick)*i/3;Text(new(X(tick)-6,bottom+16),tick.ToString(),11,muted);}
            // Discontinuous empty cohorts stay empty: never connect a line through extinction.
            for(int i=1;i<history.Count;i++) {
                var a=history[i-1].Cohort(Species,Newborn).Trait(Trait);var b=history[i].Cohort(Species,Newborn).Trait(Trait);
                if(a.Median==null||b.Median==null)continue;
                float x1=X(history[i-1].Tick),x2=X(history[i].Tick);
                DrawColoredPolygon(new[]{new Vector2(x1,bottom-a.High!.Value*ph),new Vector2(x2,bottom-b.High!.Value*ph),new Vector2(x2,bottom-b.Low!.Value*ph),new Vector2(x1,bottom-a.Low!.Value*ph)},band);
                DrawLine(new(x1,bottom-a.Median.Value*ph),new(x2,bottom-b.Median.Value*ph),line,2,true);
            }
            foreach(var e in World.Events.TakeLast(8))if(e.Tick>=minTick&&e.Tick<=maxTick)DrawLine(new(X(e.Tick),top),new(X(e.Tick),bottom),new Color(.6f,.5f,.34f,.5f),1);
            float marker=X(history[picked].Tick);DrawLine(new(marker,top),new(marker,bottom),new("9b7b45"));
            Text(new(left,Size.Y-3),"Simulation tick",11,muted);
            if(hover>=0&&hover<history.Count) {
                var h=history[hover];var t=h.Cohort(Species,Newborn).Trait(Trait);string label=t.Median is float m?$"Tick {h.Tick}: median {m:0.00}, middle 80% {t.Low:0.00}–{t.High:0.00}":$"Tick {h.Tick}: no observations";
                DrawRect(new Rect2(left,top-20,Math.Min(pw,390),22),new("f4f7f3"));Text(new(left+4,top-5),label,11);
            }
        }
    }
    public override void _GuiInput(InputEvent input) {
        if(Histogram||World==null||World.History.Count<2)return;
        if(input is InputEventMouse mouse) {
            float t=Math.Clamp((mouse.Position.X-40)/(Size.X-52),0,1);int tick=(int)(World.History[0].Tick+t*(World.History[^1].Tick-World.History[0].Tick));
            hover=Enumerable.Range(0,World.History.Count).MinBy(i=>Math.Abs(World.History[i].Tick-tick));QueueRedraw();
            if(input is InputEventMouseButton button&&button.Pressed&&button.ButtonIndex==MouseButton.Left)SelectedChanged?.Invoke(hover);
        }
    }
}
