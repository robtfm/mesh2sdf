use bevy::{
    core_pipeline::core_3d,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph},
        render_resource::{encase::private::WriteInto, *},
        renderer::{RenderContext, RenderDevice},
        RenderApp, RenderStage,
    },
};
use std::borrow::Cow;

use crate::{utils::preprocess_mesh_for_sdf, Sdf, SdfAtlas};

pub const WORKGROUP_SIZE: u32 = 8;

pub struct SdfComputePlugin;

impl Plugin for SdfComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_system_to_stage(
            CoreStage::PostUpdate,
            preprocess_sdfs.label("preprocess sdfs"),
        )
        .add_plugin(ExtractResourcePlugin::<SdfData>::default())
        .init_resource::<SdfData>();
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .init_resource::<SdfComputePipeline>()
            .add_system_to_stage(RenderStage::Queue, queue_bind_group);

        let mut render_graph = render_app.world.resource_mut::<RenderGraph>();
        let graph_3d = render_graph
            .get_sub_graph_mut(core_3d::graph::NAME)
            .unwrap();
        graph_3d.add_node("sdf_compute", SdfComputeNode::default());
        graph_3d
            .add_node_edge("sdf_compute", core_3d::graph::node::MAIN_PASS)
            .unwrap();
    }
}

#[derive(ShaderType, Clone, Debug)]
struct SdfInstanceData {
    write_position: UVec3,
    aabb_min: Vec3,
    scale: Vec3,
    block_dimensions: UVec3,
    counts: UVec3,
    block_count: u32,
}

#[derive(ShaderType, Clone, Default)]
struct SdfInstancesData {
    #[size(runtime)]
    data: Vec<SdfInstanceData>,
}

#[derive(ShaderType, Clone, Default)]
struct SdfVerticesData {
    #[size(runtime)]
    data: Vec<[Vec3; 2]>,
}

#[derive(ShaderType, Clone, Default)]
struct SdfEdgesData {
    #[size(runtime)]
    data: Vec<[Vec3; 3]>,
}

#[derive(ShaderType, Clone, Default)]
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

#[derive(Component, Clone, ExtractResource, Default)]
struct SdfData {
    bind_group: Option<BindGroup>,
    block_count: u32,
    instances: SdfInstancesData,
    vertices: SdfVerticesData,
    edges: SdfEdgesData,
    tris: SdfTrisData,
}

fn preprocess_sdfs(
    meshes: Res<Assets<Mesh>>,
    atlas: Res<SdfAtlas>,
    sdfs: Query<(&Sdf, Option<&Handle<Mesh>>, Option<&SkinnedMesh>)>,
    inverse_bindposes: Res<Assets<SkinnedMeshInverseBindposes>>,
    joint_transforms: Query<&GlobalTransform>,
    mut sdf_data: ResMut<SdfData>,
) {
    sdf_data.block_count = 0;
    sdf_data.instances.data.clear();
    sdf_data.vertices.data.clear();
    sdf_data.edges.data.clear();
    sdf_data.tris.data.clear();

    for (ent, key, aabb) in atlas.need_computing.iter() {
        let Ok((sdf, maybe_mesh, maybe_skin)) = sdfs.get(*ent) else {
            warn!("can't get sdf");
            continue;
        };

        let Some(mesh_handle) = (match sdf.mode {
            crate::SdfGenMode::FromPrimaryMesh => maybe_mesh,
            crate::SdfGenMode::Precomputed(_) => unimplemented!(),
            crate::SdfGenMode::FromCustomMesh(ref h) => Some(h),
        }) else {
            warn!("failed to get mesh handle");
            continue;
        };

        let Some(mesh) = meshes.get(mesh_handle) else {
            warn!("failed to get mesh");
            continue;
        };

        let Some(atlas_info) = atlas.page.get(key) else {
            warn!("failed to get atlas info");
            continue;
        };
        let dimensions = atlas_info.size - 1;

        let preprocessed = match maybe_skin {
            Some(skin) => {
                let Some(poses) = inverse_bindposes.get(&skin.inverse_bindposes) else {panic!("no bindposes")};

                let joints = skin
                    .joints
                    .iter()
                    .zip(poses.iter())
                    .map(|(joint_ent, pose)| {
                        joint_transforms.get(*joint_ent).unwrap().affine() * *pose
                    })
                    .collect::<Vec<_>>();
                preprocess_mesh_for_sdf(mesh, Some(&joints))
            }
            _ => preprocess_mesh_for_sdf(mesh, None),
        };

        let block_dimensions = dimensions / WORKGROUP_SIZE;
        let block_count = block_dimensions.x * block_dimensions.y * block_dimensions.z;
        sdf_data.block_count += block_count;
        sdf_data.instances.data.push(SdfInstanceData {
            block_count,
            write_position: atlas_info.position,
            aabb_min: (aabb.center - aabb.half_extents).into(),
            scale: (aabb.half_extents * 2.0 / (dimensions - 1).as_vec3a()).into(),
            block_dimensions,
            counts: UVec3::new(
                preprocessed.vertices.len() as u32,
                preprocessed.edges.len() as u32,
                preprocessed.triangles.len() as u32,
            ),
        });
        sdf_data.vertices.data.extend(
            preprocessed
                .vertices
                .into_iter()
                .map(|(v, n)| [Vec3::from(v), Vec3::from(n)]),
        );
        sdf_data.edges.data.extend(
            preprocessed
                .edges
                .into_iter()
                .map(|((v0, v1), n)| [Vec3::from(v0), Vec3::from(v1), Vec3::from(n)]),
        );
        sdf_data
            .tris
            .data
            .extend(preprocessed.triangles.into_iter().map(|tri| SdfTriData {
                a: tri.a.into(),
                b: tri.b.into(),
                c: tri.c.into(),
                plane: tri.plane.normal_d(),
                inv_area: tri.inv_area,
            }));

        // println!("[{}] preprocess: {}", *frame, block_dimensions * 8);
    }
}

fn queue_bind_group(
    atlas: Res<SdfAtlas>,
    mut sdf_data: ResMut<SdfData>,
    pipeline: Res<SdfComputePipeline>,
    gpu_images: Res<RenderAssets<Image>>,
    render_device: Res<RenderDevice>,
) {
    let Some(gpu_image) = gpu_images.get(&atlas.image) else {
        warn!("can't find gpu sdf image");
        sdf_data.bind_group = None;
        return;
    };

    if sdf_data.block_count == 0 {
        sdf_data.bind_group = None;
        return;
    }

    fn storage_buffer<T: ShaderType + WriteInto>(
        storage_data: &T,
        label: &'static str,
        render_device: &RenderDevice,
    ) -> Buffer {
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
                resource: storage_buffer(&sdf_data.instances, "sdf instances", &render_device)
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: storage_buffer(&sdf_data.vertices, "sdf vertices", &render_device)
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: storage_buffer(&sdf_data.edges, "sdf edges", &render_device)
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: storage_buffer(&sdf_data.tris, "sdf triangles", &render_device)
                    .as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&gpu_image.texture_view),
            },
        ],
    });
    sdf_data.bind_group = Some(bind_group);
    // println!("[{}] render_queue {}", *frame, sdf_data.instances.data[0].block_dimensions * 8);
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
                                ty: BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: Some(SdfInstancesData::min_size()),
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
        let sdf_data = world.resource::<SdfData>();
        let Some(bind_group) = sdf_data.bind_group.as_ref() else { return Ok(()) };
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<SdfComputePipeline>();

        // println!("running {} blocks", sdf_data.block_count);
        // let block_counts = sdf_data.instances.data.iter().map(|d| d.block_count).collect::<Vec<_>>();
        // println!("block counts: {:?}", block_counts);
        // if block_counts.iter().sum::<u32>() == 1 {
        //     println!("instance data: {:?}", sdf_data.instances.data[0]);
        // }

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, bind_group, &[]);

        pass.set_pipeline(
            pipeline_cache
                .get_compute_pipeline(pipeline.pipeline)
                .unwrap(),
        );
        pass.dispatch_workgroups(sdf_data.block_count, 1, 1);

        // println!("dispatch: {}", sdf_data.instances.data[0].block_dimensions * 8);

        Ok(())
    }
}
