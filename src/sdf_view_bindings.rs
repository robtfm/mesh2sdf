use bevy::{
    pbr::{
        MeshUniform, UserViewBindGroupLayoutEntry, UserViewBindingsEntries, UserViewBindingsShader,
        UserViewBindingsSpec,
    },
    prelude::*,
    render::{
        render_resource::{
            encase::{StorageBuffer, UniformBuffer},
            AddressMode, AsBindGroup, BindingType, BufferBindingType, BufferInitDescriptor,
            BufferUsages, FilterMode, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
            ShaderType,
        },
        renderer::RenderDevice,
    },
};

use crate::{Sdf, SdfAtlas, SdfAtlasKey};

#[derive(ShaderType, AsBindGroup)]
struct SdfViewUniform {
    ao_distances: Vec3,
    ao_sin_angle: f32,
}

#[derive(ShaderType)]
struct SdfHeader {
    transform: Mat4,
    aabb_min: Vec3,
    aabb_size: Vec3,
    atlas_position: Vec3,
    atlas_size: Vec3,
    scale: f32,
}

#[derive(ShaderType)]
struct SdfHeaders {
    #[size(runtime)]
    data: Vec<SdfHeader>,
}

pub(crate) fn add_view_bindings(app: &mut App) {
    let mut user_bindings = app
        .world
        .get_resource_or_insert_with::<UserViewBindingsSpec>(|| Default::default());
    user_bindings.layout_entries.extend([
        (
            "sdf_uniform",
            UserViewBindGroupLayoutEntry {
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(SdfViewUniform::min_size()),
                },
            },
        ),
        (
            "sdf_headers",
            UserViewBindGroupLayoutEntry {
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(SdfHeaders::min_size()),
                },
            },
        ),
        (
            "sdf_atlas",
            UserViewBindGroupLayoutEntry {
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: bevy::render::render_resource::TextureSampleType::Float {
                        filterable: true,
                    },
                    view_dimension: bevy::render::render_resource::TextureViewDimension::D3,
                    multisampled: false,
                },
            },
        ),
        (
            "sdf_sampler",
            UserViewBindGroupLayoutEntry {
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
            },
        ),
    ]);

    user_bindings.binding_shaders.push(UserViewBindingsShader {
        shader: String::from(include_str!("sdf_view_bindings.wgsl")),
        num_bindings: 5,
    });
}

pub(crate) fn queue_sdf_view_bindings(
    mut view_bindings: ResMut<UserViewBindingsEntries>,
    atlas: Res<SdfAtlas>,
    render_device: Res<RenderDevice>,
    sdfs: Query<(&Sdf, Option<&Handle<Mesh>>, &MeshUniform)>,
    mut frame: Local<u32>,
    mut sampler: Local<Option<Sampler>>,
) {
    *frame = (*frame + 1) % 1000;

    let view_uniform = SdfViewUniform {
        ao_distances: Vec3::new(0.1, 0.2, 0.3),
        ao_sin_angle: 0.5,
    };

    let byte_buffer = Vec::with_capacity(SdfViewUniform::min_size().get() as usize);
    let mut buffer = UniformBuffer::new(byte_buffer);
    buffer.write(&view_uniform).unwrap();

    let view_uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf view uniform"),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        contents: buffer.as_ref(),
    });

    let sdf_headers = sdfs.iter().filter_map(|(sdf, maybe_mesh, mesh_uniform)| {
        SdfAtlasKey::try_from_sdf(sdf, maybe_mesh)
            .and_then(|key| atlas.page.get(&key))
            .and_then(|info| {
                let scale = Transform::from_matrix(mesh_uniform.transform).scale.x;
                Some(SdfHeader {
                    transform: mesh_uniform.inverse_transpose_model.transpose(),
                    aabb_min: sdf.aabb.min().into(),
                    aabb_size: (sdf.aabb.half_extents * 2.0).into(),
                    atlas_position: info.position.as_vec3() / atlas.page.dim.as_vec3(),
                    atlas_size: (info.size - 1).as_vec3() / atlas.page.dim.as_vec3(),
                    scale,
                })
            })
    });

    // if let Some((sdf, maybe_mesh, mesh_uniform)) = sdfs.iter().nth(4) {
    //     if let Some(key) = SdfAtlasKey::try_from_sdf(sdf, maybe_mesh) {
    //         if let Some(info) = atlas.page.get(&key) {
    //             println!(
    //                 "sdf 4 is {} @ {}",
    //                 info.size - 1,
    //                 mesh_uniform.transform.w_axis.truncate()
    //             );
    //         }
    //     }
    // }

    let sdf_headers = SdfHeaders {
        data: sdf_headers.collect(),
    };

    // println!("{}", sdf_headers.data.len());

    let byte_buffer =
        Vec::with_capacity(SdfHeaders::min_size().get() as usize * sdf_headers.data.len());
    let mut buffer = StorageBuffer::new(byte_buffer);
    buffer.write(&sdf_headers).unwrap();

    let view_sdf_headers_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("sdf headers"),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        contents: buffer.as_ref(),
    });

    let sampler = sampler.get_or_insert_with(|| {
        render_device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        })
    });

    view_bindings
        .entries
        .insert("sdf_uniform", Box::new(view_uniform_buffer));
    view_bindings
        .entries
        .insert("sdf_headers", Box::new(view_sdf_headers_buffer));
    view_bindings
        .entries
        .insert("sdf_atlas", Box::new(atlas.image.clone()));
    view_bindings
        .entries
        .insert("sdf_sampler", Box::new(sampler.clone()));
}
