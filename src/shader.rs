// an easy way to make single texture / no texture materials

use bevy::{
    ecs::system::{lifetimeless::SRes, SystemParam},
    pbr::MaterialPipeline,
    prelude::*,
    reflect::TypeUuid,
    render::{
        render_asset::{PrepareAssetError, RenderAsset, RenderAssets},
        render_resource::{encase::private::WriteInto, *},
        renderer::RenderDevice,
    },
};

#[derive(Clone)]
pub struct GpuBufferedMaterial {
    pub buffer: Buffer,
    pub bind_group: BindGroup,
}

pub trait SimpleTextureSpec: Sync + Send + Clone + TypeUuid + 'static {
    type Param: SystemParam;
    type Uniform: ShaderType + WriteInto;

    fn sample_type() -> TextureSampleType {
        TextureSampleType::Float { filterable: true }
    }

    fn dimension() -> TextureViewDimension {
        TextureViewDimension::D2
    }

    fn cull_mode() -> Option<Face> {
        Some(Face::Front)
    }

    fn prepare_uniform_data(
        &self,
        param: &mut bevy::ecs::system::SystemParamItem<Self::Param>,
    ) -> Option<Self::Uniform>;
    fn texture_handle(&self) -> &Handle<Image>;

    fn alpha_mode() -> AlphaMode {
        AlphaMode::Opaque
    }
    #[allow(unused_variables)]
    fn vertex_shader(asset_server: &AssetServer) -> Option<Handle<Shader>> {
        None
    }
    #[allow(unused_variables)]
    fn fragment_shader(asset_server: &AssetServer) -> Option<Handle<Shader>> {
        None
    }
}

#[derive(Clone, Copy)]
pub struct SimpleTextureMaterial<S: SimpleTextureSpec>(pub S);

impl<S: SimpleTextureSpec> TypeUuid for SimpleTextureMaterial<S> {
    const TYPE_UUID: bevy::reflect::Uuid = <S as TypeUuid>::TYPE_UUID;
}

impl<S: SimpleTextureSpec<Param = P>, P: SystemParam> RenderAsset for SimpleTextureMaterial<S> {
    type ExtractedAsset = SimpleTextureMaterial<S>;
    type PreparedAsset = GpuBufferedMaterial;
    type Param = (
        <S as SimpleTextureSpec>::Param,
        SRes<RenderDevice>,
        SRes<MaterialPipeline<Self>>,
        SRes<RenderAssets<Image>>,
    );

    fn extract_asset(&self) -> Self::ExtractedAsset {
        println!("extract");
        self.clone()
    }

    fn prepare_asset(
        material: Self::ExtractedAsset,
        (uniform_param, render_device, material_pipeline, gpu_images): &mut bevy::ecs::system::SystemParamItem<Self::Param>,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let uniform_data = material.0.prepare_uniform_data(uniform_param);

        let uniform_data = match uniform_data {
            Some(u) => u,
            None => return Err(PrepareAssetError::RetryNextUpdate(material.clone())),
        };

        let (base_color_texture_view, base_color_sampler) = if let Some(result) = material_pipeline
            .mesh_pipeline
            .get_image_texture(gpu_images, &Some(material.0.texture_handle().clone()))
        {
            result
        } else {
            return Err(PrepareAssetError::RetryNextUpdate(material));
        };

        let byte_buffer = vec![0u8; S::Uniform::min_size().get() as usize];
        let mut buffer = encase::UniformBuffer::new(byte_buffer);
        buffer.write(&uniform_data).unwrap();

        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("material uniform buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: buffer.as_ref(),
        });

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(base_color_texture_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(base_color_sampler),
                },
            ],
            label: None,
            layout: &material_pipeline.material_layout,
        });

        Ok(GpuBufferedMaterial { buffer, bind_group })
    }
}

impl<S: SimpleTextureSpec> Material for SimpleTextureMaterial<S> {
    fn alpha_mode(_: &GpuBufferedMaterial) -> AlphaMode {
        S::alpha_mode()
    }

    fn vertex_shader(asset_server: &AssetServer) -> Option<Handle<Shader>> {
        S::vertex_shader(asset_server)
    }

    fn fragment_shader(asset_server: &AssetServer) -> Option<Handle<Shader>> {
        S::fragment_shader(asset_server)
    }

    fn bind_group(
        material: &<Self as bevy::render::render_asset::RenderAsset>::PreparedAsset,
    ) -> &bevy::render::render_resource::BindGroup {
        &material.bind_group
    }

    fn bind_group_layout(
        render_device: &bevy::render::renderer::RenderDevice,
    ) -> bevy::render::render_resource::BindGroupLayout {
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(S::Uniform::min_size()),
                    },
                    count: None,
                },
                // Base Color Texture
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        multisampled: false,
                        sample_type: S::sample_type(),
                        view_dimension: S::dimension(),
                    },
                    count: None,
                },
                // Base Color Texture Sampler
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: None,
        })
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &bevy::render::mesh::MeshVertexBufferLayout,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = S::cull_mode();
        Ok(())
    }
}

pub trait SimpleUniformSpec: Sync + Send + Clone + TypeUuid + 'static {
    type Param: SystemParam;
    type Uniform: ShaderType + WriteInto;

    fn prepare_uniform_data(
        &self,
        param: &mut bevy::ecs::system::SystemParamItem<Self::Param>,
    ) -> Option<Self::Uniform>;

    fn alpha_mode() -> AlphaMode {
        AlphaMode::Opaque
    }
    #[allow(unused_variables)]
    fn vertex_shader(asset_server: &AssetServer) -> Option<Handle<Shader>> {
        None
    }
    #[allow(unused_variables)]
    fn fragment_shader(asset_server: &AssetServer) -> Option<Handle<Shader>> {
        None
    }
}

#[derive(Clone, Copy)]
pub struct SimpleUniformMaterial<S: SimpleUniformSpec>(pub S);

impl<S: SimpleUniformSpec> TypeUuid for SimpleUniformMaterial<S> {
    const TYPE_UUID: bevy::reflect::Uuid = <S as TypeUuid>::TYPE_UUID;
}

impl<S: SimpleUniformSpec<Param = P>, P: SystemParam> RenderAsset for SimpleUniformMaterial<S> {
    type ExtractedAsset = SimpleUniformMaterial<S>;
    type PreparedAsset = GpuBufferedMaterial;
    type Param = (
        <S as SimpleUniformSpec>::Param,
        SRes<RenderDevice>,
        SRes<MaterialPipeline<Self>>,
    );

    fn extract_asset(&self) -> Self::ExtractedAsset {
        self.clone()
    }

    fn prepare_asset(
        material: Self::ExtractedAsset,
        (uniform_param, render_device, material_pipeline): &mut bevy::ecs::system::SystemParamItem<
            Self::Param,
        >,
    ) -> Result<Self::PreparedAsset, PrepareAssetError<Self::ExtractedAsset>> {
        let Some(uniform_data) = material
            .0
            .prepare_uniform_data(uniform_param) else {
                return Err(PrepareAssetError::RetryNextUpdate(material.clone()));
        };

        let byte_buffer = vec![0u8; S::Uniform::min_size().get() as usize];
        let mut buffer = encase::UniformBuffer::new(byte_buffer);
        buffer.write(&uniform_data).unwrap();

        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("material uniform buffer"),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            contents: buffer.as_ref(),
        });

        let bind_group = render_device.create_bind_group(&BindGroupDescriptor {
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: None,
            layout: &material_pipeline.material_layout,
        });

        Ok(GpuBufferedMaterial { buffer, bind_group })
    }
}

impl<S: SimpleUniformSpec> Material for SimpleUniformMaterial<S> {
    fn alpha_mode(_: &GpuBufferedMaterial) -> AlphaMode {
        S::alpha_mode()
    }

    fn vertex_shader(asset_server: &AssetServer) -> Option<Handle<Shader>> {
        S::vertex_shader(asset_server)
    }

    fn fragment_shader(asset_server: &AssetServer) -> Option<Handle<Shader>> {
        S::fragment_shader(asset_server)
    }

    fn bind_group(
        material: &<Self as bevy::render::render_asset::RenderAsset>::PreparedAsset,
    ) -> &bevy::render::render_resource::BindGroup {
        &material.bind_group
    }

    fn bind_group_layout(
        render_device: &bevy::render::renderer::RenderDevice,
    ) -> bevy::render::render_resource::BindGroupLayout {
        render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(S::Uniform::min_size()),
                },
                count: None,
            }],
            label: None,
        })
    }
}
