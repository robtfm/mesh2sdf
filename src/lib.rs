#![feature(let_else, slice_as_chunks)]
pub mod animated_aabb;
pub mod compute;
pub mod controller;
pub mod cpu;
pub mod debug_render;
mod sdf_view_bindings;
pub mod utils;

use animated_aabb::AnimatedAabbBuilder;
use atlas3d::AtlasPage;
use bevy::{
    asset::load_internal_asset,
    pbr::{queue_mesh_view_bind_groups, PBR_AMBIENT_HANDLE},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::skinning::SkinnedMesh,
        primitives::Aabb,
        view::VisibilitySystems::CheckVisibility,
        RenderApp, RenderStage,
    },
};
use compute::{SdfComputePlugin, WORKGROUP_SIZE};
use utils::create_sdf_image;

use crate::sdf_view_bindings::queue_sdf_view_bindings;

#[derive(Component, Clone)]
pub struct Sdf {
    pub mode: SdfGenMode,
    pub options: SdfOptions,
    pub aabb: Aabb,
}

impl Default for Sdf {
    fn default() -> Self {
        Self {
            mode: SdfGenMode::FromPrimaryMesh,
            options: Default::default(),
            aabb: Default::default(),
        }
    }
}

impl Sdf {
    pub fn new_scaled(scale: f32) -> Self {
        Self {
            options: SdfOptions {
                scale_multiplier: scale,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

impl ExtractComponent for Sdf {
    type Query = &'static Self;
    type Filter = ();

    fn extract_component(item: bevy::ecs::query::QueryItem<Self::Query>) -> Self {
        item.clone()
    }
}

#[derive(Clone)]
pub enum SdfGenMode {
    // generate the sdf from the mesh attached to the owning entity
    FromPrimaryMesh,
    // use a precomputed sdf texture
    Precomputed(Handle<Image>),
    // use a custom mesh to generate the sdf (can be simplified, etc)
    FromCustomMesh(Handle<Mesh>),
}

#[derive(Clone)]
pub struct SdfOptions {
    // specify the scale multiplier
    // by default, sdfs are generated with dimensions approximately matching the SdfPlugin::unit_size
    // this setting allows scaling of those dimensions on this entity for precision or speed
    pub scale_multiplier: f32,
    // buffer size (defaults to global buffer_size)
    pub buffer_size: Option<f32>,
}

impl Default for SdfOptions {
    fn default() -> Self {
        Self {
            scale_multiplier: 1.0,
            buffer_size: None,
        }
    }
}

pub struct SdfGlobalSettings {
    // size of the atlas used for storing all sdfs
    pub atlas_page_size: UVec3,
    // generated aabbs will be extended by this amount (divided by the entity's scale)
    // this should be as large as the ambient tap max distance and the maximum soft shadow cone radius
    // shadow cone radius depends on light range and cone angle/softness
    pub buffer_size: f32,
    // default sdf unit size
    pub unit_size: f32,
    // ambient occlusion distance
    pub ambient_distance: f32,
}

impl Default for SdfGlobalSettings {
    fn default() -> Self {
        Self {
            // 32mb atlas page
            atlas_page_size: UVec3::splat(200),
            buffer_size: 1.0,
            unit_size: 1.0,
            ambient_distance: 1.0,
        }
    }
}

pub struct SdfPlugin;

impl SdfPlugin {
    pub fn add_view_bindings(app: &mut App) {
        sdf_view_bindings::add_view_bindings(app)
    }
}

impl Plugin for SdfPlugin {
    fn build(&self, app: &mut App) {
        // settings
        let settings = app
            .world
            .get_resource_or_insert_with(|| SdfGlobalSettings::default());
        let page_size = settings.atlas_page_size;

        // create atlas resource
        let image = create_sdf_image(page_size);
        let image = app.world.resource_mut::<Assets<Image>>().add(image);
        app.insert_resource(SdfAtlas {
            page: AtlasPage::new(page_size),
            image,
            need_computing: Vec::new(),
        });

        // and extract it
        app.add_plugin(ExtractResourcePlugin::<SdfAtlas>::default());

        // system to generate required sdfs
        app.add_system_to_stage(
            CoreStage::PostUpdate,
            queue_sdfs.after(CheckVisibility).before("preprocess sdfs"),
        );

        // extract sdfs
        app.add_plugin(ExtractComponentPlugin::<Sdf>::default());

        // compute pass
        app.add_plugin(SdfComputePlugin);

        // add view bindings
        app.sub_app_mut(RenderApp).add_system_to_stage(
            RenderStage::Queue,
            queue_sdf_view_bindings.before(queue_mesh_view_bind_groups),
        );

        // override occlusion function
        load_internal_asset!(
            app,
            PBR_AMBIENT_HANDLE,
            "sdf_ambient.wgsl",
            Shader::from_wgsl
        );
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum SdfAtlasKey {
    Mesh(Handle<Mesh>),
    Image(Handle<Image>),
}

#[derive(Clone, ExtractResource)]
pub struct SdfAtlas {
    pub page: AtlasPage<SdfAtlasKey>,
    pub image: Handle<Image>,
    pub need_computing: Vec<(Entity, SdfAtlasKey, Aabb)>,
}

fn sdf_dim(aabb: &Aabb, unit_size: f32, buffer_size: f32) -> UVec3 {
    ((((aabb.half_extents + buffer_size) * 2.0) / unit_size) / WORKGROUP_SIZE as f32)
        .ceil()
        .as_uvec3()
        * WORKGROUP_SIZE
}

impl SdfAtlasKey {
    fn try_from_sdf(sdf: &Sdf, maybe_mesh: Option<&Handle<Mesh>>) -> Option<SdfAtlasKey> {
        Some(match &sdf.mode {
            SdfGenMode::FromPrimaryMesh => match maybe_mesh {
                Some(h) => Self::Mesh(h.clone_weak()),
                None => return None,
            },
            SdfGenMode::Precomputed(h) => Self::Image(h.clone_weak()),
            SdfGenMode::FromCustomMesh(h) => Self::Mesh(h.clone_weak()),
        })
    }
}

fn queue_sdfs(
    sdf_settings: Res<SdfGlobalSettings>,
    mut items: Query<(
        Entity,
        &mut Sdf,
        &GlobalTransform,
        &ComputedVisibility,
        &Aabb,
        Option<&SkinnedMesh>,
        Option<&Handle<Mesh>>,
    )>,
    aabb_builder: AnimatedAabbBuilder,
    mut atlas: ResMut<SdfAtlas>,
) {
    atlas.page.remove_all();
    atlas.need_computing.clear();
    for (ent, mut sdf, _g_trans, vis, aabb, maybe_skin, maybe_mesh) in items.iter_mut() {
        let Some(key) = SdfAtlasKey::try_from_sdf(&sdf, maybe_mesh) else {continue};

        let mut use_aabb = aabb.clone();

        if maybe_skin.is_some() {
            // purge previous instance of animated items (no point in clogging up the atlas)
            atlas.page.purge(&key);

            if vis.is_visible() {
                // update animated item aabbs
                use_aabb = match sdf.mode {
                    SdfGenMode::FromPrimaryMesh => aabb_builder.animated_aabb(ent).unwrap(),
                    SdfGenMode::Precomputed(_) => {
                        panic!("can't use precomputed sdf with animated meshes")
                    }
                    SdfGenMode::FromCustomMesh(ref h) => {
                        aabb_builder.animated_aabb_for_mesh(ent, h).unwrap()
                    }
                };
            }
        }

        let buffer_size = sdf.options.buffer_size.unwrap_or(sdf_settings.buffer_size);
        use_aabb.half_extents += buffer_size;

        if vis.is_visible() {
            let dims = sdf_dim(
                &use_aabb,
                sdf_settings.unit_size / sdf.options.scale_multiplier,
                buffer_size,
            );
            let res = atlas.page.insert(key.clone(), dims + 1);

            match res {
                atlas3d::Slot::New(_) => {
                    println!("queue: {}", dims);
                    atlas.need_computing.push((ent, key, use_aabb.clone()));
                    sdf.aabb = use_aabb;
                }
                atlas3d::Slot::NoFit => warn!("can't fit {} into atlas", dims + 1),
                atlas3d::Slot::Existing(_) => (),
            }
        }
    }
}
