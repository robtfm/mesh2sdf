//! A compute shader that simulates Conway's Game of Life.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::{*, encase::private::WriteInto},
        renderer::{RenderContext, RenderDevice},
        RenderApp, RenderStage, extract_component::{ExtractComponentPlugin, ExtractComponent}, mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
    },
    utils::HashMap,
};
use std::borrow::Cow;

use crate::{Sdf, preprocess_mesh_for_sdf};

const WORKGROUP_SIZE: u32 = 8;

pub struct SdfComputePlugin;

impl Plugin for SdfComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app
            .add_system_to_stage(CoreStage::First, clear_reqs)
            .add_system_to_stage(CoreStage::PostUpdate, preprocess_sdfs)
            .add_plugin(ExtractResourcePlugin::<SdfComputeRequests>::default())
            .add_plugin(ExtractComponentPlugin::<SdfPreprocessedData>::default())
            .init_resource::<SdfComputeRequests>()
        ;
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SdfComputeBindGroups>()
            .init_resource::<SdfComputePipeline>()
            .init_resource::<SdfReque>()
            .add_system_to_stage(RenderStage::Queue, queue_bind_groups);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        render_graph.add_node("sdf_compute", SdfComputeNode::default());
        render_graph
            .add_node_edge(
                "sdf_compute",
                bevy::render::main_graph::node::CAMERA_DRIVER,
            )
            .unwrap();
    }
}

fn clear_reqs(
    mut commands: Commands,
    preps: Query<Entity, With<SdfPreprocessedData>>,
    mut reqs: ResMut<SdfComputeRequests>,

) {
    reqs.0.clear();
    for preppie in preps.iter() {
        commands.entity(preppie).remove::<SdfPreprocessedData>();
    }
}

#[derive(Clone, Deref, ExtractResource, Default)]
pub struct SdfComputeRequests(pub Vec<Entity>);

#[derive(Default)]
struct SdfReque(Vec<(Entity, SdfPreprocessedData)>);


#[derive(Default)]
struct SdfComputeBindGroups(HashMap<Entity, (BindGroup, UVec3)>);

#[derive(ShaderType, Clone)]
struct SdfUniformData {
    aabb_min: Vec3,
    scale: Vec3,
}

#[derive(ShaderType, Clone)]
struct SdfVerticesData {
    #[size(runtime)]
    data: Vec<[Vec3; 2]>,
}

#[derive(ShaderType, Clone)]
struct SdfEdgesData {
    #[size(runtime)]
    data: Vec<[Vec3; 3]>,
}

#[derive(ShaderType, Clone)]
struct SdfTrisData {
    #[size(runtime)]
    data: Vec<SdfTriData>,
}

#[derive(ShaderType, Clone)]
struct SdfTriData {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    plane: Vec4,
    inv_area: f32,
}

#[derive(Component, Clone)]
struct SdfPreprocessedData{
    image: Handle<Image>,
    dimensions: UVec3,
    uniform: SdfUniformData,
    vertices: SdfVerticesData,
    edges: SdfEdgesData,
    tris: SdfTrisData,
}

impl ExtractComponent for SdfPreprocessedData {
    type Query = &'static Self;
    type Filter = ();

    fn extract_component(item: bevy::ecs::query::QueryItem<Self::Query>) -> Self {
        item.clone()
    }
}

fn preprocess_sdfs(
    mut commands: Commands,
    meshes: Res<Assets<Mesh>>,
    images: Res<Assets<Image>>,
    reqs: Res<SdfComputeRequests>,
    sdfs: Query<(&Sdf, &Handle<Mesh>, Option<&SkinnedMesh>)>,
    inverse_bindposes: Res<Assets<SkinnedMeshInverseBindposes>>,
    joint_transforms: Query<&GlobalTransform>,
) {
    for ent in reqs.0.iter() {
        let Ok((sdf, mesh, maybe_skin)) = sdfs.get(*ent) else {
            warn!("can't get sdf + mesh handle");
            continue;
        };

        let Some(mesh) = meshes.get(mesh) else {
            warn!("failed to get mesh");
            continue;
        };

        let Some(image) = images.get(&sdf.image) else {
            warn!("can't find raw image");
            continue;
        };
        let dimensions = image.texture_descriptor.size;
        let dimensions = UVec3::new(dimensions.width, dimensions.height, dimensions.depth_or_array_layers);

        let uniform = SdfUniformData {
            aabb_min: (sdf.aabb.center - sdf.aabb.half_extents).into(),
            scale: (sdf.aabb.half_extents * 2.0 / (dimensions - 1).as_vec3a()).into(),
        };

        let preprocessed = match maybe_skin {
            Some(skin) => {
                let Some(poses) = inverse_bindposes.get(&skin.inverse_bindposes) else {panic!("no bindposes")};

                let joints = skin.joints.iter().zip(poses.iter()).map(|(joint_ent, pose)| {
                    joint_transforms.get(*joint_ent).unwrap().compute_affine() * *pose
                }).collect::<Vec<_>>();
                preprocess_mesh_for_sdf(mesh, Some(&joints))
            }
            _ => preprocess_mesh_for_sdf(mesh, None)
        };

        let vertices = SdfVerticesData {
            data: preprocessed.vertices.into_iter().map(|(v,n)| [Vec3::from(v), Vec3::from(n)]).collect(),
        };

        let edges = SdfEdgesData {
            data: preprocessed.edges.into_iter().map(|((v0,v1), n)| [Vec3::from(v0), Vec3::from(v1), Vec3::from(n)]).collect(),
        };

        let tris = SdfTrisData {
            data: preprocessed.triangles.into_iter().map(|tri| SdfTriData { a: tri.a.into(), b: tri.b.into(), c: tri.c.into(), plane: tri.plane.normal_d(), inv_area: tri.inv_area }).collect(),
        };

        commands.entity(*ent).insert(SdfPreprocessedData{
            image: sdf.image.clone(),
            dimensions,
            uniform,
            vertices,
            edges,
            tris,
        });
    }
}

fn queue_bind_groups(
    pipeline: Res<SdfComputePipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    sdfs: Query<(Entity, &SdfPreprocessedData)>,
    mut sdf_bindgroups: ResMut<SdfComputeBindGroups>,
    render_device: Res<RenderDevice>,
) {
    sdf_bindgroups.0.clear();
    for (ent, preproc) in sdfs.iter() {
        let Some(gpu_image) = gpu_images.get(&preproc.image) else {
            warn!("can't find gpu image");
            continue;
        };

        fn uniform_buffer<T: ShaderType + WriteInto>(uniform_data: &T, label: &'static str, render_device: &RenderDevice) -> Buffer {
            let byte_buffer = vec![0u8; T::min_size().get() as usize];
            let mut buffer = encase::UniformBuffer::new(byte_buffer);
            buffer.write(uniform_data).unwrap();
    
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                contents: buffer.as_ref(),
            })
        }

        fn storage_buffer<T: ShaderType + WriteInto>(storage_data: &T, label: &'static str, render_device: &RenderDevice) -> Buffer {
            let byte_buffer = vec![0u8; T::min_size().get() as usize];
            let mut buffer = encase::StorageBuffer::new(byte_buffer);
            buffer.write(storage_data).unwrap();
    
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some(label),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                contents: buffer.as_ref(),
            })
        }

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer(&preproc.uniform, "sdf uniform", &render_device).as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: storage_buffer(&preproc.vertices, "sdf vertices", &render_device).as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: storage_buffer(&preproc.edges, "sdf edges", &render_device).as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: storage_buffer(&preproc.tris, "sdf triangles", &render_device).as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&gpu_image.texture_view),
                },
            ],
        });
        sdf_bindgroups.0.insert(ent, (bind_group, preproc.dimensions));
    }
}

pub struct SdfComputePipeline {
    bind_group_layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for SdfComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let bind_group_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        // sdf header
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: Some(SdfUniformData::min_size()),
                            },
                            count: None,
                        },
                        // vertices
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(SdfVerticesData::min_size()),
                            },
                            count: None,
                        },
                        // edges
                        BindGroupLayoutEntry {
                            binding: 2,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(SdfEdgesData::min_size()),
                            },
                            count: None,
                        },
                        // tris
                        BindGroupLayoutEntry {
                            binding: 3,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(SdfTrisData::min_size()),
                            },
                            count: None,
                        },
                        // output
                        BindGroupLayoutEntry {
                            binding: 4,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::StorageTexture {
                                access: StorageTextureAccess::WriteOnly,
                                format: TextureFormat::R32Float,
                                view_dimension: TextureViewDimension::D3,
                            },
                            count: None,
                        },                    
                    ],
                });

        let shader = world
            .resource::<AssetServer>()
            .load("shader/compute_sdf.wgsl");
        let mut pipeline_cache = world.resource_mut::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: Some(vec![bind_group_layout.clone()]),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("calc"),
        });

        SdfComputePipeline {
            bind_group_layout,
            pipeline,
        }
    }
}

#[derive(Default)]
struct SdfComputeNode;

impl render_graph::Node for SdfComputeNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = world.resource::<SdfComputeBindGroups>().0.values().collect::<Vec<_>>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<SdfComputePipeline>();

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        for (bind_group, dims) in bind_groups {
            pass.set_bind_group(0, bind_group, &[]);

            let dims = (dims.as_vec3() / WORKGROUP_SIZE as f32).ceil().as_uvec3();
            // println!("dims: {}", dims);

            pass.set_pipeline(pipeline_cache
                .get_compute_pipeline(pipeline.pipeline)
                .unwrap()
            );
            pass.dispatch(dims.x, dims.y, dims.z);
        }

        Ok(())
    }
}
