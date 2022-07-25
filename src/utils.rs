use std::collections::BTreeMap;

use bevy::{
    math::Vec3A,
    prelude::*,
    render::{
        mesh::VertexAttributeValues,
        primitives::Plane,
        render_resource::{
            AddressMode, Extent3d, FilterMode, SamplerDescriptor, TextureDimension, TextureUsages,
        },
        texture::ImageSampler,
    },
    utils::FloatOrd,
};

#[derive(PartialEq, Clone, Copy, Debug)]
struct OrderedVec(Vec3A);

impl Eq for OrderedVec {}
impl Ord for OrderedVec {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        FloatOrd(self.0.x)
            .cmp(&FloatOrd(other.0.x))
            .then(FloatOrd(self.0.y).cmp(&FloatOrd(other.0.y)))
            .then(FloatOrd(self.0.z).cmp(&FloatOrd(other.0.z)))
    }
}

impl PartialOrd for OrderedVec {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

#[derive(Debug)]
pub struct TriData {
    pub a: Vec3A,
    pub b: Vec3A,
    pub c: Vec3A,
    pub inv_area: f32,
    pub plane: Plane,
}

pub struct PreprocessedMeshData {
    // pub aabb: Aabb,
    pub vertices: Vec<(Vec3A, Vec3A)>,
    pub edges: Vec<((Vec3A, Vec3A), Vec3A)>,
    pub triangles: Vec<TriData>,
}

pub fn preprocess_mesh_for_sdf(mesh: &Mesh, joints: Option<&[Mat4]>) -> PreprocessedMeshData {
    let Some(VertexAttributeValues::Float32x3(values)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        panic!("bad mesh");
    };

    let weight_with_joints = |v: Vec3, index: usize| -> Vec3 {
        let Some(VertexAttributeValues::Float32x4(joint_weights)) = mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT) else {panic!("bad joint weights!")};
        let Some(VertexAttributeValues::Uint16x4(joint_indexes)) = mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX) else {panic!("bad joint indexes!")};
        let joints = joints.unwrap();
        let indexes = joint_indexes[index];
        let weights = joint_weights[index];
        let mat = joints[indexes[0] as usize] * weights[0]
            + joints[indexes[1] as usize] * weights[1]
            + joints[indexes[2] as usize] * weights[2]
            + joints[indexes[3] as usize] * weights[3];
        let res = mat * v.extend(1.0);
        res.truncate() / res.w
    };

    let weight = |v: Vec3, index: usize| -> Vec3 {
        if joints.is_some() {
            weight_with_joints(v, index)
        } else {
            v
        }
    };

    let values: Vec<Vec3> = match mesh.indices() {
        Some(ix) => ix
            .iter()
            .map(|ix| weight(Vec3::from(values[ix]), ix))
            .collect(),
        None => values
            .iter()
            .enumerate()
            .map(|(ix, v)| weight(Vec3::from(*v), ix))
            .collect(),
    };

    let mut vertices = BTreeMap::<OrderedVec, Vec3A>::new();
    let mut edges = BTreeMap::<(OrderedVec, OrderedVec), Vec3A>::new();
    let mut triangles = Vec::<TriData>::new();

    for tri in values.chunks_exact(3) {
        let a = OrderedVec(tri[0].into());
        let b = OrderedVec(tri[1].into());
        let c = OrderedVec(tri[2].into());

        let normal = (b.0 - a.0).cross(c.0 - b.0).normalize();

        // sort
        let mut sorted = vec![a, b, c];
        sorted.sort();
        let mut iter = sorted.into_iter();
        let (a, b, c) = (
            iter.next().unwrap(),
            iter.next().unwrap(),
            iter.next().unwrap(),
        );

        let ab_len = (b.0 - a.0).length();
        let ac_len = (c.0 - a.0).length();
        let bc_len = (c.0 - b.0).length();

        let a_angle = tri_angle(bc_len, ab_len, ac_len);
        let b_angle = tri_angle(ac_len, ab_len, bc_len);
        let c_angle = tri_angle(ab_len, ac_len, bc_len);

        *vertices.entry(a).or_default() += normal * a_angle;
        *vertices.entry(b).or_default() += normal * b_angle;
        *vertices.entry(c).or_default() += normal * c_angle;

        *edges.entry((a, b)).or_default() += normal;
        *edges.entry((a, c)).or_default() += normal;
        *edges.entry((b, c)).or_default() += normal;

        let plane = Plane::new(normal.extend(-(a.0).dot(normal)));
        let inv_area = (b.0 - a.0).cross(c.0 - a.0).dot(plane.normal()).recip();

        triangles.push(TriData {
            a: a.0,
            b: b.0,
            c: c.0,
            inv_area,
            plane,
        });
    }

    fn tri_angle(opp: f32, a: f32, b: f32) -> f32 {
        ((a * a + b * b - opp * opp) / (2.0 * a * b)).acos()
    }

    // let (min, max) = vertices.keys().fold((Vec3A::splat(f32::MAX), Vec3A::splat(f32::MIN)), |(cur_min, cur_max), v| {
    //     (cur_min.min(v.0), cur_max.max(v.0))
    // });

    PreprocessedMeshData {
        // aabb: Aabb::from_min_max(Vec3::from(min), Vec3::from(max)),
        vertices: vertices.into_iter().map(|(ov, n)| (ov.0, n)).collect(),
        edges: edges
            .into_iter()
            .map(|((ov0, ov1), n)| ((ov0.0, ov1.0), n))
            .collect(),
        triangles,
    }
}

pub fn create_sdf_image(dimension: UVec3) -> Image {
    let mut image = Image::new_fill(
        Extent3d {
            width: dimension.x,
            height: dimension.y,
            depth_or_array_layers: dimension.z,
        },
        TextureDimension::D3,
        &[0; 4],
        bevy::render::render_resource::TextureFormat::R32Float,
    );

    image.sampler_descriptor = ImageSampler::Descriptor(SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..Default::default()
    });

    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;

    image
}
