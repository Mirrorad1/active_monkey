using Godot;
using Wilding.Simulation;
namespace Wilding.Game;

public partial class TerrainView:Node3D {
    readonly List<MeshInstance3D> chunks=new();
    MultiMeshInstance3D? trunks,canopies,rocks;
    readonly ShaderMaterial material=new(){Shader=new Shader{Code=@"shader_type spatial;
render_mode cull_disabled;
uniform bool food_overlay=false;
uniform sampler2D food_map : filter_nearest;
varying vec2 tile_uv;
void vertex(){tile_uv=(MODEL_MATRIX*vec4(VERTEX,1.0)).xz/64.0;}
void fragment(){ALBEDO=COLOR.rgb;if(food_overlay){float value=texture(food_map,tile_uv).r;ALBEDO=mix(vec3(.69,.60,.36),vec3(.28,.58,.36),value);}ROUGHNESS=1.0;}
"}};
    ImageTexture? foodTexture;
    public int FoodTick { get; private set; }=-1;
    public int Overlay;
    public static float Elevation(World w,float x,float z)=>w.IsLand((int)x,(int)z)?w.Tiles[w.Index((int)x,(int)z)].Height*.48f:0;
    public void Rebuild(World w) {
        material.SetShaderParameter("food_overlay",Overlay==1);UpdateFood(w);
        foreach(var c in chunks){RemoveChild(c);c.QueueFree();}chunks.Clear();
        var treePositions=new List<(Vector3,Color)>();var rockPositions=new List<(Vector3,Color)>();
        for(int cz=0;cz<4;cz++)for(int cx=0;cx<4;cx++) {
            var surface=new SurfaceTool();surface.Begin(Mesh.PrimitiveType.Triangles);
            void Quad(Vector3 a,Vector3 b,Vector3 c,Vector3 d,Vector3 normal,Color color) {
                foreach(var v in new[]{a,b,c,a,c,d}){surface.SetNormal(normal);surface.SetColor(color);surface.AddVertex(v);}
            }
            for(int z=cz*16;z<(cz+1)*16;z++)for(int x=cx*16;x<(cx+1)*16;x++) {
                var t=w.Tiles[w.Index(x,z)];if(t.Height==0)continue;float h=t.Height*.48f;
                Color color=t.Height==1?new("d7cd9d"):t.Height>4?new("a4ad9c"):new Color(.49f+t.Height*.018f,.65f+t.Height*.012f,.41f+t.Height*.012f);
                if(Overlay==2)color=x<27?new("97abbc"):z>34?new("bca77e"):new("b49aaa");
                Quad(new(x,h,z),new(x+1,h,z),new(x+1,h,z+1),new(x,h,z+1),Vector3.Up,color);
                float west=x==0?-.6f:Elevation(w,x-1,z),east=x==63?-.6f:Elevation(w,x+1,z),north=z==0?-.6f:Elevation(w,x,z-1),south=z==63?-.6f:Elevation(w,x,z+1);
                if(west<h)Quad(new(x,west,z),new(x,h,z),new(x,h,z+1),new(x,west,z+1),Vector3.Left,color.Darkened(.1f));
                if(east<h)Quad(new(x+1,east,z+1),new(x+1,h,z+1),new(x+1,h,z),new(x+1,east,z),Vector3.Right,color.Darkened(.05f));
                if(north<h)Quad(new(x+1,north,z),new(x+1,h,z),new(x,h,z),new(x,north,z),Vector3.Forward,color.Darkened(.06f));
                if(south<h)Quad(new(x,south,z+1),new(x,h,z+1),new(x+1,h,z+1),new(x+1,south,z+1),Vector3.Back,color.Darkened(.14f));
                if(t.Forest){float jitter=((x*17+z*31)%11)*.014f;treePositions.Add((new(x+.5f,h,z+.5f),new Color(.29f+jitter,.46f+jitter,.30f)));}
                if(t.Refuge)rockPositions.Add((new(x+.5f,h,z+.5f),new("89988e")));
            }
            var mesh=surface.Commit();if(mesh==null)continue;mesh.SurfaceSetMaterial(0,material);var chunk=new MeshInstance3D{Mesh=mesh,CastShadow=GeometryInstance3D.ShadowCastingSetting.Off};AddChild(chunk);chunks.Add(chunk);
        }
        ReplaceBatch(ref trunks,treePositions,new Vector3(.12f,.5f,.12f),new Vector3(0,.25f,0),new("7b6747"));
        ReplaceBatch(ref canopies,treePositions,new Vector3(.72f,.95f,.72f),new Vector3(0,.94f,0),null);
        ReplaceBatch(ref rocks,rockPositions,new Vector3(.9f,.65f,.9f),new Vector3(0,.325f,0),null);
    }
    public void UpdateFood(World world) {
        using var image=Image.CreateEmpty(World.Width,World.Width,false,Image.Format.R8);
        for(int z=0;z<World.Width;z++)for(int x=0;x<World.Width;x++){var t=world.Tiles[world.Index(x,z)];float ratio=t.Capacity>0?t.Food/t.Capacity:0;image.SetPixel(x,z,new Color(ratio,0,0));}
        if(foodTexture==null){foodTexture=ImageTexture.CreateFromImage(image);material.SetShaderParameter("food_map",foodTexture);}else foodTexture.Update(image);
        FoodTick=world.Tick;
    }
    void ReplaceBatch(ref MultiMeshInstance3D? node,List<(Vector3,Color)> items,Vector3 scale,Vector3 offset,Color? fixedColor) {
        if(node!=null){RemoveChild(node);node.QueueFree();}
        var mesh=new BoxMesh{Size=Vector3.One};mesh.Material=new StandardMaterial3D{VertexColorUseAsAlbedo=true,Roughness=1};
        var mm=new MultiMesh{TransformFormat=MultiMesh.TransformFormatEnum.Transform3D,UseColors=true,Mesh=mesh,InstanceCount=items.Count};
        for(int i=0;i<items.Count;i++){mm.SetInstanceTransform(i,new Transform3D(Basis.Identity.Scaled(scale),items[i].Item1+offset));mm.SetInstanceColor(i,fixedColor??items[i].Item2);}
        node=new MultiMeshInstance3D{Multimesh=mm,CastShadow=GeometryInstance3D.ShadowCastingSetting.Off};AddChild(node);
    }
}
