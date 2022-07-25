use bevy::{
    ecs::system::SystemParam,
    prelude::*,
    render::{
        mesh::{
            skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
            VertexAttributeValues,
        },
        primitives::Aabb,
    },
};

/// generate an aabb for the current animation state of the mesh
/// example usage:
///
/// fn update_aabbs(
///     mut to_update: Query<(Entity, &mut Aabb)>,
///     aabb_builder: AnimatedAabbBuilder,
/// ) {
///     for (ent, mut aabb) in to_update.iter_mut() {
///         aabb = aabb_builder.animated_aabb(ent).unwrap();
///     }
/// }
///
#[derive(SystemParam)]
pub struct AnimatedAabbBuilder<'w, 's> {
    meshes: Res<'w, Assets<Mesh>>,
    inverse_bindposes: Res<'w, Assets<SkinnedMeshInverseBindposes>>,
    mesh_query: Query<'w, 's, (&'static Handle<Mesh>, &'static SkinnedMesh)>,
    global_transforms: Query<'w, 's, &'static GlobalTransform>,
}

impl<'w, 's> AnimatedAabbBuilder<'w, 's> {
    pub fn animated_aabb(&self, ent: Entity) -> Option<Aabb> {
        let (mesh_handle, _) = self.mesh_query.get(ent).ok()?;
        self.animated_aabb_for_mesh(ent, mesh_handle)
    }

    pub fn animated_aabb_for_mesh(&self, ent: Entity, mesh_handle: &Handle<Mesh>) -> Option<Aabb> {
        let (_, skin) = self.mesh_query.get(ent).ok()?;
        let mesh = self.meshes.get(mesh_handle)?;
        let poses = self.inverse_bindposes.get(&skin.inverse_bindposes)?;
        let VertexAttributeValues::Float32x3(values) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)? else {return None};
        let VertexAttributeValues::Float32x4(joint_weights) = mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT)? else {return None};
        let VertexAttributeValues::Uint16x4(joint_indexes) = mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX)? else {return None};

        let joints = skin
            .joints
            .iter()
            .zip(poses.iter())
            .map(|(joint_ent, pose)| {
                self.global_transforms.get(*joint_ent).unwrap().affine() * *pose
            })
            .collect::<Vec<_>>();

        let weight = |v: Vec3, index: usize| -> Vec3 {
            let indexes = joint_indexes[index];
            let weights = joint_weights[index];
            let mat = joints[indexes[0] as usize] * weights[0]
                + joints[indexes[1] as usize] * weights[1]
                + joints[indexes[2] as usize] * weights[2]
                + joints[indexes[3] as usize] * weights[3];
            let res = mat * v.extend(1.0);
            res.truncate() / res.w
        };

        let (minimum, maximum) = match mesh.indices() {
            Some(indices) => indices.iter().fold(
                (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                |(cur_min, cur_max), ix| {
                    let vertex = weight(Vec3::from(values[ix]), ix);
                    (cur_min.min(vertex), cur_max.max(vertex))
                },
            ),
            None => values.iter().enumerate().fold(
                (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                |(cur_min, cur_max), (ix, v)| {
                    let vertex = weight(Vec3::from(*v), ix);
                    (cur_min.min(vertex), cur_max.max(vertex))
                },
            ),
        };

        if minimum.max_element() != std::f32::MAX && maximum.min_element() != std::f32::MIN {
            return Some(Aabb::from_min_max(minimum, maximum));
        }

        None
    }
}
