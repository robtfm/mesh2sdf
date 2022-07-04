use crate::{
    shader::{SimpleTextureMaterial, SimpleTextureSpec},
    Sdf,
};
use bevy::{
    prelude::*,
    reflect::TypeUuid,
    render::{
        primitives::Aabb,
        render_resource::{ShaderType, TextureSampleType, TextureViewDimension},
    },
};

pub struct SdfRenderPlugin;

pub type SdfMaterial = SimpleTextureMaterial<SdfMaterialSpec>;

pub enum SdfRenderBounds {
    Aabb,
    FullScreen,
    ExtendedAabb(Vec3),
}

impl Plugin for SdfRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(MaterialPlugin::<SdfMaterial>::default());
        app.add_system(gen_sdf_render_mesh);
    }
}

#[derive(Clone, TypeUuid)]
#[uuid = "8f83afc2-8543-40d9-b8ec-fbdb11051ebf"]
pub struct SdfMaterialSpec {
    pub image: Handle<Image>,
    pub aabb: Aabb,
    pub min_step_size: f32,
    pub hit_threshold: f32,
    pub max_step_count: usize,
    pub base_color: Color,
    pub hit_color: Color,
    pub step_color: Color,
    pub distance_color: Color,
}

impl Default for SdfMaterialSpec {
    fn default() -> Self {
        Self {
            image: Default::default(),
            aabb: Default::default(),
            min_step_size: Default::default(),
            hit_threshold: Default::default(),
            max_step_count: Default::default(),
            base_color: Color::rgba_linear(0.0, 0.0, 0.0, 1.0),
            hit_color: Color::rgba_linear(1.0, 0.0, 0.0, 0.0),
            step_color: Color::rgba_linear(0.0, 1.0, 0.0, 0.0),
            distance_color: Color::rgba_linear(0.0, 0.0, 1.0, 0.0),
        }
    }
}

#[derive(ShaderType)]
pub struct SdfMaterialUniformData {
    aabb_min: Vec3,
    aabb_extents: Vec3,
    base_color: Vec4,
    hit_color: Vec4,
    step_color: Vec4,
    distance_color: Vec4,
    min_step_size: f32,
    hit_threshold: f32,
    max_step_count: u32,
}

impl SimpleTextureSpec for SdfMaterialSpec {
    type Param = ();
    type Uniform = SdfMaterialUniformData;

    fn alpha_mode() -> AlphaMode {
        AlphaMode::Blend
    }
    fn texture_handle(&self) -> &Handle<Image> {
        &self.image
    }
    fn sample_type() -> TextureSampleType {
        TextureSampleType::Float { filterable: true }
    }

    fn dimension() -> TextureViewDimension {
        TextureViewDimension::D3
    }

    fn cull_mode() -> Option<bevy::render::render_resource::Face> {
        None
    }

    fn fragment_shader(asset_server: &AssetServer) -> Option<Handle<Shader>> {
        asset_server.watch_for_changes().unwrap();
        Some(asset_server.load("shader/render_sdf.wgsl"))
    }

    fn prepare_uniform_data(&self, _: &mut Self::Param) -> Option<Self::Uniform> {
        println!("prep");

        Some(SdfMaterialUniformData {
            aabb_min: (self.aabb.center - self.aabb.half_extents).into(),
            aabb_extents: (self.aabb.half_extents * 2.0).into(),
            min_step_size: self.min_step_size,
            hit_threshold: self.hit_threshold,
            max_step_count: self.max_step_count as u32,
            base_color: self.base_color.as_linear_rgba_f32().into(),
            hit_color: self.hit_color.as_linear_rgba_f32().into(),
            step_color: self.step_color.as_linear_rgba_f32().into(),
            distance_color: self.distance_color.as_linear_rgba_f32().into(),
        })
    }
}

fn gen_sdf_render_mesh(
    mut commands: Commands,
    q: Query<(Entity, &Handle<SdfMaterial>, &Sdf), Without<Handle<Mesh>>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for (ent, _render, sdf) in q.iter() {
        let min = sdf.aabb.center - sdf.aabb.half_extents;
        let max = sdf.aabb.center + sdf.aabb.half_extents;
        let mesh = shape::Box {
            min_x: min.x,
            max_x: max.x,
            min_y: min.y,
            max_y: max.y,
            min_z: min.z,
            max_z: max.z,
        }
        .into();
        let mesh = meshes.add(mesh);
        commands.entity(ent).insert_bundle((
            mesh,
            Transform::default(),
            GlobalTransform::default(),
            Visibility::default(),
            ComputedVisibility::default(),
        ));
        println!("+ mesh");
    }
}
