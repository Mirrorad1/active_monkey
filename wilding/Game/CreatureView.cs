using Godot;
using Wilding.Simulation;
namespace Wilding.Game;

public partial class CreatureView:Node3D {
    readonly MultiMeshInstance3D[] batches=new MultiMeshInstance3D[2];
    readonly ShaderMaterial[] materials=new ShaderMaterial[2];
    public override void _Ready() {
        for(int species=0;species<2;species++) {
            var shader=new Shader{Code=@"shader_type spatial;
render_mode unshaded, cull_disabled, depth_prepass_alpha;
uniform sampler2D atlas : source_color, filter_nearest;
uniform float sim_time = 0.0;
varying vec4 detail;
void vertex() {
detail=INSTANCE_CUSTOM;
MODELVIEW_MATRIX=VIEW_MATRIX*mat4(INV_VIEW_MATRIX[0]*length(MODEL_MATRIX[0].xyz),INV_VIEW_MATRIX[1]*length(MODEL_MATRIX[1].xyz),INV_VIEW_MATRIX[2]*length(MODEL_MATRIX[2].xyz),MODEL_MATRIX[3]);
}
void fragment() {
vec2 uv=UV;
if(detail.x>0.5)uv.x=1.0-uv.x;
uv.x=(uv.x+mod(floor(sim_time+detail.y),2.0))*0.5;
vec4 pixel=texture(atlas,uv);if(pixel.a<0.5)discard;
ALBEDO=pixel.rgb*COLOR.rgb;ALPHA=1.0;
}"};
            materials[species]=new ShaderMaterial{Shader=shader};materials[species].SetShaderParameter("atlas",SpriteTexture(species==1));
            var mesh=new QuadMesh{Size=new Vector2(1.05f,.98f),Material=materials[species]};
            var mm=new MultiMesh{TransformFormat=MultiMesh.TransformFormatEnum.Transform3D,UseColors=true,UseCustomData=true,Mesh=mesh,InstanceCount=8192,VisibleInstanceCount=0};
            batches[species]=new MultiMeshInstance3D{Multimesh=mm,CastShadow=GeometryInstance3D.ShadowCastingSetting.Off,CustomAabb=new Aabb(new(-5,-5,-5),new(80,30,80))};AddChild(batches[species]);
        }
    }
    public void Refresh(World world,bool lineages=false) {
        int[] counts={0,0};
        foreach(var c in world.Creatures) {
            int s=(int)c.Species,index=counts[s]++;if(index>=8192)continue;
            float size=(.64f+c.Traits.Size*.4f)*(c.Age<60?.72f:1);
            var transform=new Transform3D(Basis.Identity.Scaled(Vector3.One*size),new Vector3(c.X,TerrainView.Elevation(world,c.X,c.Z)+.45f*size,c.Z));
            batches[s].Multimesh.SetInstanceTransform(index,transform);
            Color color=lineages?new[]{new Color("b9cce0"),new Color("f2deb2"),new Color("ddb8cd")}[c.Lineage%3]:new Color(1,1,1);
            batches[s].Multimesh.SetInstanceColor(index,color);batches[s].Multimesh.SetInstanceCustomData(index,new Color(MathF.Cos(c.Heading)>0?1:0,c.Id%7,0,1));
        }
        for(int s=0;s<2;s++){batches[s].Multimesh.VisibleInstanceCount=Math.Min(counts[s],8192);materials[s].SetShaderParameter("sim_time",world.Tick*.22f);}
    }
    static ImageTexture SpriteTexture(bool hunter) {
        var image=Image.CreateEmpty(56,26,false,Image.Format.Rgba8);image.Fill(Colors.Transparent);
        for(int frame=0;frame<2;frame++) {
            void R(int x,int y,int width,int height,string color){image.FillRect(new Rect2I(frame*28+x,y,width,height),new Color(color));}
            const string body="bc9764",cream="f3e7cc",dark="493e39",foot="635242",fox="b9683f";
            if(hunter) {
                R(20,10,4,5,fox);R(23,8,3,5,fox);R(25,7,3,4,cream);R(8,11,13,7,fox);R(10,10,8,1,fox);R(7,14,8,4,cream);R(5,8,7,8,fox);R(3,8,8,6,fox);R(2,11,5,3,cream);R(1,11,2,2,dark);R(4,4,3,5,fox);R(9,5,3,4,fox);R(5,5,1,3,dark);R(10,6,1,2,dark);R(4,9,1,1,dark);R(9+frame,18,2,4,dark);R(18-frame,18,2,4,dark);
            }else{
                R(8,10,13,7,body);R(10,9,8,1,body);R(8,15,11,2,cream);R(6,7,5,8,body);R(3,5,8,5,body);R(1,7,5,3,body);R(4,2,2,4,body);R(9,3,2,3,body);R(6,0,1,4,foot);R(5,1,3,1,foot);R(2,8,1,1,dark);R(5,6,1,1,dark);R(20,8,4,3,body);R(21,10,3,2,cream);R(11,11,1,1,cream);R(15,12,1,1,cream);R(18,11,1,1,cream);R(13,14,1,1,cream);R(9+frame,17,2,5,foot);R(18-frame,17,2,5,foot);R(8+frame,21,3,1,dark);R(17-frame,21,3,1,dark);
            }
        }
        return ImageTexture.CreateFromImage(image);
    }
}
