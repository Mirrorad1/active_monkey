using Godot;
using System.Diagnostics;
using System.Globalization;
using Wilding.Simulation;
namespace Wilding.Game;

public partial class Sandbox:Control {
    World world=new(42,1000);
    TerrainView terrain=new();CreatureView creatureView=new();Camera3D camera=null!;SubViewport viewport=null!;
    Control worldArea=null!;ChartView trend=new(),distribution=new(){Histogram=true};
    Label counts=null!,clock=null!,notice=null!,readout=null!,performance=null!;Button pause=null!;
    OptionButton population=null!,trait=null!,cohort=null!,overlay=null!;HSlider scrub=null!;Label scrubLabel=null!;
    readonly Dictionary<string,Button> toolButtons=new();
    Brush brush=Brush.Observe;Species? introduce;
    bool paused=true,chartsVisible=true,dragging,smoke;int speed=1,overlayId;float yaw=-.65f,pitch=.78f,zoom=49;Vector3 focus=new(32,0,32);
    double accumulator,uiTimer,simMs,smokeTimer;int terrainRevision=-1;PanelContainer chartPanel=null!;
    readonly Color ink=new("29484a"),paper=new("f4f7f3"),sub=new("657d78"),edge=new("c9d8cf");
    public override void _Ready() {
        Name="Wilding";SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);BuildUI();BuildScene();RefreshAll();
        smoke=OS.GetCmdlineUserArgs().Contains("--smoke");
        if(smoke){overlayId=1;overlay.Select(1);terrain.Overlay=1;for(int i=0;i<150;i++)world.Step();RefreshAll();paused=false;pause.Text="Ⅱ Pause";}
    }
    StyleBoxFlat Box(Color color,int radius=10,int padding=12) => new(){BgColor=color,CornerRadiusTopLeft=radius,CornerRadiusTopRight=radius,CornerRadiusBottomLeft=radius,CornerRadiusBottomRight=radius,ContentMarginLeft=padding,ContentMarginRight=padding,ContentMarginTop=padding,ContentMarginBottom=padding};
    Label L(string text,int size=14,Color? color=null) {var l=new Label{Text=text};l.AddThemeFontSizeOverride("font_size",size);l.AddThemeColorOverride("font_color",color??ink);return l;}
    Button B(string text,Action action) {var b=new Button{Text=text,MouseDefaultCursorShape=CursorShape.PointingHand};b.AddThemeStyleboxOverride("normal",Box(paper,7,8));b.AddThemeStyleboxOverride("hover",Box(new("e0eae2"),7,8));b.AddThemeStyleboxOverride("pressed",Box(new("c3d8cb"),7,8));b.AddThemeColorOverride("font_color",ink);b.AddThemeFontSizeOverride("font_size",13);b.Pressed+=()=>action();return b;}
    PanelContainer Panel(Control content,int padding=12){var p=new PanelContainer();p.AddThemeStyleboxOverride("panel",Box(paper,11,padding));p.AddChild(content);return p;}
    void BuildUI() {
        Theme=new Theme{DefaultFontSize=14};Theme.SetColor("font_color","Label",ink);Theme.SetColor("font_color","Button",ink);Theme.SetColor("font_color","OptionButton",ink);
        Theme.SetStylebox("normal","OptionButton",Box(new("e2ebe4"),6,8));Theme.SetStylebox("hover","OptionButton",Box(new("d8e6dc"),6,8));Theme.SetStylebox("pressed","OptionButton",Box(new("c8ded0"),6,8));Theme.SetStylebox("panel","PopupMenu",Box(paper,7,6));Theme.SetColor("font_color","PopupMenu",ink);Theme.SetColor("font_hover_color","PopupMenu",ink);
        var root=new VBoxContainer{Name="Layout"};root.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);root.AddThemeConstantOverride("separation",0);AddChild(root);
        var header=new HBoxContainer();header.AddThemeConstantOverride("separation",16);var hp=Panel(header,13);hp.CustomMinimumSize=new(0,56);root.AddChild(hp);
        header.AddChild(L("❧  Wilding",24));header.AddChild(L("Alder island",13,sub));header.AddChild(B("Evolution charts",()=>{chartsVisible=!chartsVisible;chartPanel.Visible=chartsVisible;}));
        var spacer=new Control{SizeFlagsHorizontal=SizeFlags.ExpandFill};header.AddChild(spacer);
        header.AddChild(B("Save",Save));header.AddChild(B("Load",Load));header.AddChild(B("Fork",Fork));header.AddChild(B("Export CSV",Export));header.AddChild(B("New island",()=>{world=new World((int)(Time.GetUnixTimeFromSystem()%100000),1000);scrub.Value=scrub.MaxValue;RefreshAll();Tell("New seeded island; paused. Save the previous world before replacing it.");paused=true;pause.Text="▶ Play";}));
        worldArea=new Control{Name="WorldArea",SizeFlagsVertical=SizeFlags.ExpandFill,CustomMinimumSize=new(0,320),MouseFilter=MouseFilterEnum.Ignore};root.AddChild(worldArea);
        var host=new SubViewportContainer{Stretch=true,MouseFilter=MouseFilterEnum.Stop};host.SetAnchorsAndOffsetsPreset(LayoutPreset.FullRect);worldArea.AddChild(host);viewport=new SubViewport{Size=new(1280,500),HandleInputLocally=false,RenderTargetUpdateMode=SubViewport.UpdateMode.Always};host.AddChild(viewport);host.GuiInput+=WorldInput;
        var seasonBox=new VBoxContainer();seasonBox.AddChild(L("Alder island",17));clock=L("Tick 0  ·  oldest lineage G0",12,sub);seasonBox.AddChild(clock);var season=Panel(seasonBox);season.Position=new(18,18);worldArea.AddChild(season);
        var tools=new VBoxContainer();tools.AddThemeConstantOverride("separation",4);
        void Tool(string name,Brush value,Species? species=null){var button=B(name,()=>{brush=value;introduce=species;foreach(var pair in toolButtons)pair.Value.Modulate=pair.Key==name?new Color("b8d3c4"):Colors.White;Tell(species!=null?$"Click land to introduce a {species}; existing creature traits are unchanged.":$"{name}: click or drag on the world. Right-drag rotates, wheel zooms, WASD pans.");});toolButtons[name]=button;tools.AddChild(button);}
        Tool("Observe",Brush.Observe);Tool("Forest",Brush.Forest);Tool("Water",Brush.Water);Tool("Refuge",Brush.Refuge);Tool("Food",Brush.Food);Tool("Raise land",Brush.Raise);Tool("+ Grazers",Brush.Observe,Species.Grazer);Tool("+ Hunters",Brush.Observe,Species.Hunter);
        var tp=Panel(tools,6);tp.Position=new(18,93);worldArea.AddChild(tp);
        var right=new HBoxContainer();overlay=new OptionButton();foreach(var text in new[]{"Natural","Food availability","Lineages"})overlay.AddItem(text);overlay.ItemSelected+=id=>{overlayId=(int)id;terrain.Overlay=overlayId;terrain.Rebuild(world);creatureView.Refresh(world,overlayId==2);};right.AddChild(overlay);right.AddChild(B("↶",()=>{yaw-=.3f;UpdateCamera();}));right.AddChild(B("↷",()=>{yaw+=.3f;UpdateCamera();}));right.AddChild(B("Whole island",()=>{focus=new(32,0,32);zoom=49;UpdateCamera();}));
        var rp=Panel(right,6);rp.SetAnchorsAndOffsetsPreset(LayoutPreset.TopRight);rp.GrowHorizontal=GrowDirection.Begin;rp.Position=new(-420,18);rp.CustomMinimumSize=new(400,0);worldArea.AddChild(rp);
        var popBox=new VBoxContainer();popBox.AddChild(L("Island life",16));counts=L("",15);popBox.AddChild(counts);popBox.AddChild(L("Real births, deaths and inherited traits",11,sub));popBox.AddChild(B("Watch grazers",()=>Focus(Species.Grazer)));popBox.AddChild(B("Watch hunters",()=>Focus(Species.Hunter)));
        var pp=Panel(popBox);pp.SetAnchorsAndOffsetsPreset(LayoutPreset.BottomRight);pp.GrowHorizontal=GrowDirection.Begin;pp.GrowVertical=GrowDirection.Begin;pp.Position=new(-280,-200);pp.CustomMinimumSize=new(260,0);worldArea.AddChild(pp);
        var footer=new HBoxContainer();footer.AddThemeConstantOverride("separation",8);root.AddChild(Panel(footer,7));pause=B("▶ Play",()=>{paused=!paused;pause.Text=paused?"▶ Play":"Ⅱ Pause";});footer.AddChild(pause);
        foreach(int s in new[]{1,5,20}){int captured=s;footer.AddChild(B($"{s}×",()=>{speed=captured;Tell($"Speed {speed}×; actual tick rate is limited by the available frame budget.");}));}
        performance=L("",12,sub);footer.AddChild(performance);notice=L("Click Play to begin. Right-drag rotates; wheel zooms; WASD pans.",12,sub);notice.SizeFlagsHorizontal=SizeFlags.ExpandFill;notice.TextOverrunBehavior=TextServer.OverrunBehavior.TrimEllipsis;footer.AddChild(notice);
        BuildCharts(root);
    }
    void BuildCharts(VBoxContainer root) {
        var content=new VBoxContainer();content.AddThemeConstantOverride("separation",5);chartPanel=Panel(content,14);chartPanel.CustomMinimumSize=new(0,290);root.AddChild(chartPanel);
        var bar=new HBoxContainer();bar.AddThemeConstantOverride("separation",12);content.AddChild(bar);bar.AddChild(L("Evolution, at a glance",18));
        population=new OptionButton();population.AddItem("Grazers");population.AddItem("Hunters");bar.AddChild(population);
        trait=new OptionButton();foreach(string text in new[]{"Hearing sensitivity","Movement speed","Body size"})trait.AddItem(text);bar.AddChild(trait);
        cohort=new OptionButton();cohort.AddItem("Living creatures");cohort.AddItem("Newborns / interval");bar.AddChild(cohort);
        population.ItemSelected+=_=>UpdateCharts();trait.ItemSelected+=_=>UpdateCharts();cohort.ItemSelected+=_=>UpdateCharts();
        var stretch=new Control{SizeFlagsHorizontal=SizeFlags.ExpandFill};bar.AddChild(stretch);scrubLabel=L("Latest",12,sub);bar.AddChild(scrubLabel);
        scrub=new HSlider{MinValue=0,MaxValue=0,Step=1,CustomMinimumSize=new(140,0)};bar.AddChild(scrub);scrub.ValueChanged+=_=>UpdateCharts();bar.AddChild(B("Live",()=>{scrub.Value=scrub.MaxValue;UpdateCharts();}));
        var charts=new HBoxContainer{SizeFlagsVertical=SizeFlags.ExpandFill};charts.AddThemeConstantOverride("separation",28);content.AddChild(charts);charts.AddChild(trend);charts.AddChild(distribution);
        trend.SelectedChanged=index=>{scrub.Value=index;UpdateCharts();};
        var legend=new HBoxContainer();legend.AddChild(L("Green line: median    Shading: middle 80%    Histogram: selected cohort vs founder genotypes",11,sub));content.AddChild(legend);readout=L("",12);content.AddChild(readout);
    }
    void BuildScene() {
        var scene=new Node3D();viewport.AddChild(scene);scene.AddChild(terrain);scene.AddChild(creatureView);
        var environment=new WorldEnvironment{Environment=new Godot.Environment{BackgroundMode=Godot.Environment.BGMode.Color,BackgroundColor=new("b1d5da"),AmbientLightSource=Godot.Environment.AmbientSource.Color,AmbientLightColor=new("e1ecdf"),AmbientLightEnergy=.78f,TonemapMode=Godot.Environment.ToneMapper.Linear}};scene.AddChild(environment);
        var sun=new DirectionalLight3D{LightEnergy=.8f,RotationDegrees=new(-50,-35,0),ShadowEnabled=false};scene.AddChild(sun);
        var water=new MeshInstance3D{Mesh=new PlaneMesh{Size=new Vector2(86,86)},Position=new(32,-.06f,32),MaterialOverride=new StandardMaterial3D{AlbedoColor=new("a3cdd3"),Roughness=1,ShadingMode=BaseMaterial3D.ShadingModeEnum.Unshaded}};scene.AddChild(water);
        camera=new Camera3D{Projection=Camera3D.ProjectionType.Orthogonal,Current=true,Far=250};scene.AddChild(camera);UpdateCamera();
    }
    void UpdateCamera(){camera.Size=zoom;camera.Position=focus+new Vector3(MathF.Sin(yaw)*MathF.Cos(pitch),MathF.Sin(pitch),MathF.Cos(yaw)*MathF.Cos(pitch))*85;camera.LookAt(focus);}
    void Focus(Species species) {var c=world.Creatures.FirstOrDefault(c=>c.Species==species);if(c==null){Tell($"No {species.ToString().ToLowerInvariant()} population remains.");return;}focus=new(c.X,TerrainView.Elevation(world,c.X,c.Z),c.Z);zoom=16;UpdateCamera();}
    void Tell(string text){notice.Text=text;}
    void RefreshAll(){terrain.Rebuild(world);terrainRevision=world.Revision;creatureView.Refresh(world,overlayId==2);UpdateStats();}
    void UpdateStats() {
        int grazers=world.Creatures.Count(c=>c.Species==Species.Grazer),hunters=world.Creatures.Count-grazers;
        counts.Text=$"Grazers    {grazers:N0}\nHunters    {hunters:N0}\nBirths       {world.TotalBirths:N0}\nDeaths     {world.TotalDeaths:N0}";
        clock.Text=$"Tick {world.Tick:N0}  ·  maximum lineage G{world.MaxGeneration}";
        performance.Text=$"{simMs:0.00} ms/tick  ·  {Engine.GetFramesPerSecond()} fps";
        if(world.CapBirthsBlocked>0)performance.Text+=$"  ·  cap blocked {world.CapBirthsBlocked:N0} births";
        bool live=scrub.Value>=scrub.MaxValue;double old=scrub.Value;scrub.MaxValue=Math.Max(0,world.History.Count-1);scrub.SetValueNoSignal(live?scrub.MaxValue:Math.Min(old,scrub.MaxValue));UpdateCharts();
    }
    void UpdateCharts() {
        int index=Math.Clamp((int)scrub.Value,0,Math.Max(0,world.History.Count-1));
        foreach(var chart in new[]{trend,distribution}){chart.World=world;chart.Species=(Species)population.Selected;chart.Trait=trait.Selected;chart.Newborn=cohort.Selected==1;chart.Selected=index;chart.QueueRedraw();}
        if(world.History.Count==0)return;var sample=world.History[index];var c=sample.Cohort((Species)population.Selected,cohort.Selected==1);var t=c.Trait(trait.Selected);scrubLabel.Text=$"Tick {sample.Tick}";
        string value=t.Median is float m?$"median {m:0.00}  ·  middle 80% {t.Low:0.00}–{t.High:0.00}":"no observations";
        readout.Text=$"{c.Count:N0} {(cohort.Selected==1?"newborns in preceding 20 ticks":"living creatures")}  ·  {value}  ·  {world.Events.LastOrDefault()?.Text}";
    }
    public override void _Process(double delta) {
        if(!paused){accumulator=Math.Min(8,accumulator+delta*10*speed);int steps=0;long start=Stopwatch.GetTimestamp();while(accumulator>=1&&steps<8){world.Step();accumulator-=1;steps++;}if(steps>0)simMs=Stopwatch.GetElapsedTime(start).TotalMilliseconds/steps;creatureView.Refresh(world,overlayId==2);}
        if(terrainRevision!=world.Revision){terrain.Rebuild(world);terrainRevision=world.Revision;}
        uiTimer+=delta;if(uiTimer>.25){uiTimer=0;UpdateStats();if(overlayId==1&&terrain.FoodTick!=world.Tick)terrain.UpdateFood(world);}
        Vector3 pan=Vector3.Zero;if(Input.IsPhysicalKeyPressed(Key.W))pan.Z-=1;if(Input.IsPhysicalKeyPressed(Key.S))pan.Z+=1;if(Input.IsPhysicalKeyPressed(Key.A))pan.X-=1;if(Input.IsPhysicalKeyPressed(Key.D))pan.X+=1;
        if(pan!=Vector3.Zero){focus+=pan.Rotated(Vector3.Up,yaw)*(float)delta*zoom*.4f;focus.X=Math.Clamp(focus.X,0,64);focus.Z=Math.Clamp(focus.Z,0,64);UpdateCamera();}
        if(smoke){smokeTimer+=delta;if(smokeTimer>2){smoke=false;CompleteSmoke();}}
    }
    void CompleteSmoke() {
        try {
            if(world.Tick<=150)throw new Exception("Play did not advance the simulation clock");
            if(terrain.FoodTick<world.Tick-5)throw new Exception("Food overlay did not track live resource depletion");
            paused=true;pause.Text="▶ Play";
            var land=world.FindLand();brush=Brush.Refuge;EditAt(camera.UnprojectPosition(new Vector3(land.X+.5f,TerrainView.Elevation(world,land.X,land.Z),land.Z+.5f)));
            if(!world.Tiles[world.Index(land.X,land.Z)].Refuge)throw new Exception("Terrain picking/painting failed");
            brush=Brush.Observe;float oldZoom=zoom;WorldInput(new InputEventMouseButton{ButtonIndex=MouseButton.WheelUp,Pressed=true});if(zoom>=oldZoom)throw new Exception("Zoom input failed");zoom=49;UpdateCamera();
            string saved=world.ToJson();Save();world.Step();Load();if(world.ToJson()!=saved)throw new Exception("UI save/load failed");System.IO.File.Delete(SavePath);
            cohort.Select(1);UpdateCharts();if(!trend.Newborn||!distribution.Newborn)throw new Exception("Newborn chart selection failed");cohort.Select(0);RefreshAll();
            // The previous rendered frame is already available after two seconds.
            // Do not await FramePostDraw while macOS may suppress redraws for an
            // unfocused test window: that can prevent the smoke run from exiting.
            if(DisplayServer.GetName()!="headless")GetViewport().GetTexture().GetImage().SavePng(ProjectSettings.GlobalizePath("res://verification/smoke.png"));
            string before=world.ToJson();var fork=World.FromJson(before);for(int i=0;i<30;i++){world.Step();fork.Step();}if(world.ToJson()!=fork.ToJson())throw new Exception("UI snapshot replay diverged");
            GD.Print($"WILDING_SMOKE_OK ticks={world.Tick} creatures={world.Creatures.Count} history={world.History.Count}");GetTree().Quit();
        }catch(Exception e){GD.PushError(e.ToString());GetTree().Quit(1);}
    }
    void WorldInput(InputEvent input) {
        if(input is InputEventMouseButton b) {
            if(b.ButtonIndex==MouseButton.WheelUp&&b.Pressed){zoom=Math.Max(9,zoom*.9f);UpdateCamera();}
            if(b.ButtonIndex==MouseButton.WheelDown&&b.Pressed){zoom=Math.Min(85,zoom*1.1f);UpdateCamera();}
            if(b.ButtonIndex==MouseButton.Right)dragging=b.Pressed;
            if(b.ButtonIndex==MouseButton.Left&&b.Pressed)EditAt(b.Position);
        }
        if(input is InputEventMouseMotion motion&&dragging){yaw-=motion.Relative.X*.008f;pitch=Math.Clamp(pitch+motion.Relative.Y*.005f,.35f,1.2f);UpdateCamera();}
        if(input is InputEventMouseMotion m&&(m.ButtonMask&MouseButtonMask.Left)!=0&&brush!=Brush.Observe&&introduce==null&&m.Relative.Length()>1)EditAt(m.Position);
    }
    void EditAt(Vector2 mouse) {
        var ray=camera.ProjectRayOrigin(mouse);var direction=camera.ProjectRayNormal(mouse);
        for(float distance=0;distance<200;distance+=.25f) {
            Vector3 p=ray+direction*distance;if(p.X<0||p.Z<0||p.X>=64||p.Z>=64)continue;
            if(p.Y>TerrainView.Elevation(world,p.X,p.Z))continue;
            int x=(int)p.X,z=(int)p.Z;
            if(introduce is Species species){int added=0;for(int i=0;i<5;i++)if(world.AddCreature(species,x+.2f+i*.12f,z+.5f,new Traits(.45f,.45f,.45f),i%3)!=null)added++;world.Events.Add(new(world.Tick,$"Introduced {added} {species}s"));Tell($"Introduced {added} creatures with provided founder traits.");UpdateStats();creatureView.Refresh(world,overlayId==2);}
            else if(brush==Brush.Observe){var c=world.Creatures.MinBy(c=>(c.X-p.X)*(c.X-p.X)+(c.Z-p.Z)*(c.Z-p.Z));if(c!=null&&(new Vector2(c.X-p.X,c.Z-p.Z)).Length()<2)Tell($"{c.Species} #{c.Id} · lineage {c.Lineage+1} · G{c.Generation} · energy {c.Energy:0.0} · hearing {c.Traits.Hearing:0.00} · speed {c.Traits.Speed:0.00} · size {c.Traits.Size:0.00}");else Tell($"Habitat {x}, {z} · food {world.Tiles[world.Index(x,z)].Food:0.00}");}
            else{world.Paint(x,z,brush,2);RefreshAll();Tell($"{brush} painted at {x}, {z}. Water edits relocate affected creatures to the nearest land.");}break;
        }
    }
    string SavePath=>ProjectSettings.GlobalizePath(OS.GetCmdlineUserArgs().Contains("--smoke")?"user://verification-snapshot.json":"user://alder-island.json");
    void Save(){try{System.IO.File.WriteAllText(SavePath,world.ToJson());Tell($"Saved: {SavePath}");}catch(Exception e){Tell($"Save failed: {e.Message}");}}
    void Load(){try{world=World.FromJson(System.IO.File.ReadAllText(SavePath));paused=true;pause.Text="▶ Play";accumulator=0;RefreshAll();Tell("World restored, including random state; paused.");}catch(Exception e){Tell($"Load failed: {e.Message}");}}
    void Fork(){try{string path=ProjectSettings.GlobalizePath($"user://fork-{world.Tick}-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");System.IO.File.WriteAllText(path,world.ToJson());world=World.FromJson(world.ToJson());RefreshAll();Tell($"Continuing an independent branch; original saved to {path}");}catch(Exception e){Tell($"Fork failed: {e.Message}");}}
    void Export(){try{string path=ProjectSettings.GlobalizePath("user://traits.csv");using var writer=new System.IO.StreamWriter(path);writer.WriteLine("tick,species,cohort,trait,n,median,p10,p90");foreach(var s in world.History)foreach(Species sp in Enum.GetValues<Species>())foreach(bool newborn in new[]{false,true})for(int t=0;t<3;t++){var c=s.Cohort(sp,newborn);var v=c.Trait(t);string N(float? f)=>f?.ToString("R",CultureInfo.InvariantCulture)??"";writer.WriteLine($"{s.Tick},{sp},{(newborn?"newborn":"living")},{new[]{"hearing","speed","size"}[t]},{c.Count},{N(v.Median)},{N(v.Low)},{N(v.High)}");}Tell($"Exported actual sampled traits: {path}");}catch(Exception e){Tell($"Export failed: {e.Message}");}}
}
