use crate::{queue_sdfs, Sdf, SdfAtlas, SdfAtlasKey};
use bevy::{
    prelude::*,
    reflect::TypeUuid,
    render::render_resource::{AsBindGroup, ShaderRef},
    utils::HashMap,
};

pub struct SdfRenderPlugin;

pub enum SdfRenderBounds {
    Aabb,
    FullScreen,
    ExtendedAabb(Vec3),
}

impl Plugin for SdfRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(MaterialPlugin::<SdfMaterial>::default());
        app.add_system_to_stage(CoreStage::PostUpdate, update_sdf_render.after(queue_sdfs));
    }
}

#[derive(Component)]
pub struct SdfRender {
    pub entity: Entity,
    pub base_color: Color,
    pub hit_color: Color,
    pub step_color: Color,
    pub distance_color: Color,
    pub min_step_size: f32,
    pub hit_threshold: f32,
    pub max_step_count: u32,
}

#[derive(Clone, TypeUuid, AsBindGroup)]
#[uuid = "8f83afc2-8543-40d9-b8ec-fbdb11051ebf"]
pub struct SdfMaterial {
    #[uniform(0)]
    pub position: Vec3,
    #[uniform(0)]
    pub size: Vec3,
    #[uniform(0)]
    pub scale: f32,
    #[uniform(0)]
    pub aabb_min: Vec3,
    #[uniform(0)]
    pub aabb_extents: Vec3,
    #[uniform(0)]
    pub base_color: Color,
    #[uniform(0)]
    pub hit_color: Color,
    #[uniform(0)]
    pub step_color: Color,
    #[uniform(0)]
    pub distance_color: Color,
    #[uniform(0)]
    pub min_step_size: f32,
    #[uniform(0)]
    pub hit_threshold: f32,
    #[uniform(0)]
    pub max_step_count: u32,
}

impl Material for SdfMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Path("shader/render_sdf.wgsl".into())
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline<Self>,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::render::mesh::MeshVertexBufferLayout,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

fn update_sdf_render(
    mut commands: Commands,
    atlas: Res<SdfAtlas>,
    q: Query<(Entity, &SdfRender)>,
    sdf: Query<(&Sdf, Option<&Handle<Mesh>>, &GlobalTransform)>,
    changed_scale: Query<(&Handle<SdfMaterial>, &GlobalTransform), Changed<GlobalTransform>>,
    vis: Query<&ComputedVisibility>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SdfMaterial>>,
) {
    let lookup: HashMap<_, _> = atlas
        .need_computing
        .iter()
        .map(|(_ent, key, aabb)| ((key, aabb)))
        .collect();

    for (ent, render) in q.iter() {
        let Ok((sdf, maybe_mesh, g_trans)) = sdf.get(render.entity) else {continue};
        let key = SdfAtlasKey::try_from_sdf(sdf, maybe_mesh).unwrap();

        if let Some(&aabb) = lookup.get(&key) {
            let min = aabb.min();
            let max = aabb.max();
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
            let atlas_info = atlas.page.get(&key).unwrap(); // we only add to the compute queue if we are in the atlas
            println!(
                "[{:?}] render: {} @ {}",
                ent,
                atlas_info.size - 1,
                atlas_info.position
            );

            let material = SdfMaterial {
                position: atlas_info.position.as_vec3() / atlas.page.dim.as_vec3(),
                size: (atlas_info.size - 1).as_vec3() / atlas.page.dim.as_vec3(),
                aabb_min: Vec3::from(min),
                aabb_extents: Vec3::from(max - min),
                base_color: render.base_color,
                hit_color: render.hit_color,
                step_color: render.step_color,
                distance_color: render.distance_color,
                min_step_size: render.min_step_size,
                hit_threshold: render.hit_threshold,
                max_step_count: render.max_step_count,
                scale: g_trans.to_scale_rotation_translation().0.x,
            };
            let material = materials.add(material);

            let computed_vis = vis.get(render.entity).cloned().unwrap_or_default();
            commands.entity(ent).insert_bundle((
                mesh,
                material,
                Transform::default(),
                GlobalTransform::default(),
                Visibility::default(),
                computed_vis,
            ));
        }
    }

    for (mat_handle, g_trans) in changed_scale.iter() {
        if let Some(mat) = materials.get_mut(mat_handle) {
            mat.scale = g_trans.to_scale_rotation_translation().0.x;
        }
    }
}
