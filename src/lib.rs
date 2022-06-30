#![feature(let_else, slice_as_chunks, bool_to_option)]
use bevy::{prelude::*, render::{primitives::{Aabb, Plane}, render_resource::{Extent3d, TextureDimension}, mesh::{VertexAttributeValues, PrimitiveTopology}}, math::Vec3A, utils::FloatOrd};

pub struct Mesh2Sdf;

impl Plugin for Mesh2Sdf {
    fn build(&self, app: &mut App) {
        app.add_system(generate_sdfs);
    }
}

#[derive(Component)]
pub struct Sdf {
    pub image: Handle<Image>,
}

#[derive(Component)]
pub struct AnalyticSdf {
    func: SdfFunc,
    dimension: UVec3,
}

#[derive(Component)]
pub struct MeshSdf {
    dimensions: UVec3,
}

pub struct SdfFunc(Box<dyn Fn(Vec3) -> f32 + Sync + Send + 'static>);

impl SdfFunc {
    pub fn sphere(origin: Vec3, radius: f32) -> SdfFunc {
        Self(Box::new(move |p: Vec3| p.distance(origin) - radius))
    }

    pub fn union(funcs: impl Iterator<Item=SdfFunc>) -> SdfFunc {
        let funcs: Vec<_> = funcs.collect();
        Self(Box::new(move |p: Vec3| funcs.iter().fold(f32::MAX, |min, f| f32::min(min, f.0(p)))))
    }
}

#[derive(Debug)]
struct TriData {
    a: Vec3A,
    b: Vec3A,
    c: Vec3A,
}

pub fn create_sdf_from_mesh_gpu(
    mesh: &Mesh,
    aabb: &Aabb,
    dimension: UVec3,
) -> Image {
    assert!(
        matches!(mesh.primitive_topology(), PrimitiveTopology::TriangleList),
        "`sdf generation can only work on `TriangleList`s"
    );

    let Some(VertexAttributeValues::Float32x3(values)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        panic!("bad mesh");
    };

    let values: Vec<[f32;3]> = match mesh.indices() {
        Some(ix) => {
            ix.iter().map(|ix| { values[ix]}).collect()
        }
        None => {
            values.iter().copied().collect()
        }
    };

    let scale = aabb.half_extents / dimension.as_vec3a();
    let shift = scale * 0.2;

    let _triangles: Vec<TriData> = values.chunks_exact(3).map(|abc| {
        let a = Vec3A::from(abc[0]);
        let b = Vec3A::from(abc[1]);
        let c = Vec3A::from(abc[2]);

        // push
        let center = (a + b + c) / 3.0;
        let [a, b, c] = [a, b, c].map(|v| {
            let offset = center - v;
            v + offset * f32::min(0.5, shift.dot(offset.abs()) / offset.length_squared())
        });

        TriData { a, b, c }
    }).collect();

    Image::new_fill(
        Extent3d{ width: dimension.x, height: dimension.y, depth_or_array_layers: dimension.z },
        TextureDimension::D3,
        &f32::MAX.to_le_bytes(),
        bevy::render::render_resource::TextureFormat::R32Float,
    )
}

pub fn create_sdf_from_mesh_cpu(
    mesh: &Mesh,
    aabb: &Aabb,
    dimension: UVec3,
) -> Image {
    assert!(
        matches!(mesh.primitive_topology(), PrimitiveTopology::TriangleList),
        "`sdf generation can only work on `TriangleList`s"
    );

    let Some(VertexAttributeValues::Float32x3(values)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        panic!("bad mesh");
    };

    let values: Vec<[f32;3]> = match mesh.indices() {
        Some(ix) => {
            ix.iter().map(|ix| { values[ix]}).collect()
        }
        None => {
            values.iter().copied().collect()
        }
    };

    let scale = aabb.half_extents / dimension.as_vec3a();
    let shift = scale * 0.2;

    #[derive(Debug)]
    struct TriData {
        index: usize,
        a: Vec3A,
        b: Vec3A,
        c: Vec3A,
        ab_dir: Vec3A,
        ab_len: f32,
        ac_dir: Vec3A,
        ac_len: f32,
        bc_dir: Vec3A,
        bc_len: f32,
        inv_area: f32,
        plane: Plane,
    }

    let triangles: Vec<TriData> = values.chunks_exact(3).enumerate().map(|(index, abc)| {
        let a = Vec3A::from(abc[0]);
        let b = Vec3A::from(abc[1]);
        let c = Vec3A::from(abc[2]);
        let normal = (b-a).cross(c-b).normalize();

        let ab_dir = (b-a).normalize();
        let ac_dir = (c-a).normalize();
        let bc_dir = (c-b).normalize();
        let ab_len = (b-a).length();
        let ac_len = (c-a).length();
        let bc_len = (c-b).length();

        let plane = Plane::new(normal.extend(-a.dot(normal)));

        let inv_area = (b-a).cross(c-a).dot(plane.normal()).recip();
        // push
        let center = (a + b + c) / 3.0;
        let [a, b, c] = [a, b, c].map(|v| {
            let offset = center - v;
            v + offset * f32::min(0.5, shift.dot(offset.abs()) / offset.length_squared())
        });

        TriData { index, a, b, c, ab_dir, ab_len, ac_dir, ac_len, bc_dir, bc_len, inv_area, plane }
    }).collect();

    let distance_to_line_sq = |point: Vec3A, line_a: Vec3A, line: Vec3A, cap: f32| -> f32 {
        let intercept = f32::clamp((point - line_a).dot(line), 0.0, cap);
        point.distance_squared(line_a + intercept * line)
    };

    let compute_distance = |point: Vec3A, upper_bound: f32| -> f32 {
        let (dist_sq, outside, _ix) = triangles.iter().fold((upper_bound * upper_bound, true, None), |(best, outside, ix), tri| {
            // signed distance to plane = (nx * px + ny * py + nz * pz + d) / n.length()
            let distance_to_plane = tri.plane.normal_d().dot(point.extend(1.0));
            let distance_to_plane_sq = distance_to_plane * distance_to_plane;
            if distance_to_plane_sq > best {
                (best, outside, ix)
            } else {
                let point_on_plane = point + distance_to_plane * tri.plane.normal();
                // barycentric coords
                let u = (tri.c-tri.b).cross(point_on_plane-tri.b).dot(tri.plane.normal()) * tri.inv_area;
                let v = (tri.a-tri.c).cross(point_on_plane-tri.c).dot(tri.plane.normal()) * tri.inv_area;
                let w = 1.0 - u - v;
                let dist_sq = match (u.is_sign_positive(), v.is_sign_positive(), w.is_sign_positive()) {
                    (true, true, true) => {
                        // projected point is inside the triangle
                        distance_to_plane_sq
                    }
                    (false, true, true) => {
                        // nearest point is on line bc or is b or c
                        distance_to_line_sq(point, tri.b, tri.bc_dir, tri.bc_len)
                    }
                    (true, false, true) => {
                        // nearest point is on line ac or is a or c
                        distance_to_line_sq(point, tri.a, tri.ac_dir, tri.ac_len)
                    }
                    (true, true, false) => {
                        // nearest point is on line ab or is a or b
                        distance_to_line_sq(point, tri.a, tri.ab_dir, tri.ab_len)
                    }
                    (true, false, false) => {
                        // nearest point is a
                        (point-tri.a).length_squared()
                    }
                    (false, true, false) => {
                        // nearest point is b
                        (point-tri.b).length_squared()
                    }
                    (false, false, true) => {
                        // nearest point is c
                        (point-tri.c).length_squared()
                    }
                    (false, false, false) => unreachable!()
                };

                if dist_sq > best {
                    (best, outside, ix)
                } else {
                    (dist_sq, distance_to_plane >= 0.0, Some(tri.index))
                }
            }
        });

        // println!("tri used: {:?}", triangles[_ix.unwrap()]);

        if outside {
            dist_sq.sqrt()
        } else {
            -dist_sq.sqrt()
        }
    };

    let mut data: Vec<u8> = Vec::new();
    data.resize((4 * dimension.x * dimension.y * dimension.z) as usize, 0);
    let mut chunks = data.as_chunks_mut::<4>().0.iter_mut();

    for x in 0..dimension.x {
        for y in 0..dimension.y {
            for z in 0..dimension.z {
                let point = aabb.min() + scale * ((UVec3::new(x,y,z) * 2) + 1).as_vec3a();

                let dist = compute_distance(point, f32::MAX);
                let chunk = chunks.next().unwrap();
                chunk.copy_from_slice(&dist.to_le_bytes());
            }
        }
    }

    Image::new(
        Extent3d{ width: dimension.x, height: dimension.y, depth_or_array_layers: dimension.z },
        TextureDimension::D3,
        data,
        bevy::render::render_resource::TextureFormat::R32Float,
    )
}

fn create_sdf_from_fn(
    func: &SdfFunc,
    aabb: &Aabb,
    dimension: UVec3,
) -> Image {
    let scale = Vec3::from(aabb.half_extents) / dimension.as_vec3();

    let mut data: Vec<u8> = Vec::new();
    data.resize((4 * dimension.x * dimension.y * dimension.z) as usize, 0);
    let mut chunks = data.as_chunks_mut::<4>().0.iter_mut();

    for x in 0..dimension.x {
        for y in 0..dimension.y {
            for z in 0..dimension.z {
                let point = Vec3::from(aabb.min()) + scale * ((UVec3::new(x,y,z) * 2) + 1).as_vec3();
                let dist = func.0(point);
                let chunk = chunks.next().unwrap();
                chunk.copy_from_slice(&dist.to_le_bytes());
            }
        }
    }

    Image::new(
        Extent3d{ width: dimension.x, height: dimension.y, depth_or_array_layers: dimension.z },
        TextureDimension::D3,
        data,
        bevy::render::render_resource::TextureFormat::R32Float,
    )
}

fn generate_sdfs(
    mut commands: Commands,
    new_by_mesh: Query<(Entity, &Handle<Mesh>, &Aabb, &MeshSdf), Without<Sdf>>,
    new_by_anal: Query<(Entity, &Aabb, &AnalyticSdf), Without<Sdf>>,
    meshes: Res<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
) {
    for (ent, mesh_handle, aabb, autosdf) in new_by_mesh.iter() {
        if let Some(mesh) = meshes.get(mesh_handle) {
            let image = create_sdf_from_mesh_cpu(mesh, aabb, autosdf.dimensions);
            let image = images.add(image);
            commands.entity(ent).insert(Sdf { image });
        }
    }

    for (ent, aabb, anal) in new_by_anal.iter() {
        let image = create_sdf_from_fn(&anal.func, aabb, anal.dimension);
        let image = images.add(image);
        commands.entity(ent).insert(Sdf { image });
    }
}

#[cfg(test)]
mod test {
    use bevy::{prelude::*, render::{mesh::PrimitiveTopology, primitives::Aabb}, math::Vec3A};

    use crate::{create_sdf_from_mesh_cpu, SdfFunc, create_sdf_from_fn};

    #[test]
    fn icosphere_sdf() {
        let icosphere = shape::Icosphere{ radius: 6.5, subdivisions: 10 };
        let mesh = Mesh::from(icosphere);
        println!("vertices: {}", mesh.count_vertices());
        let aabb = Aabb{center: Vec3A::ZERO, half_extents: Vec3A::splat(10.0) };
        let dimensions = UVec3::new(25, 25, 25);
        let sdf = create_sdf_from_mesh_cpu(&mesh, &aabb, dimensions);

        // let scale = aabb.half_extents / dimensions.as_vec3a();

        for y in 0..dimensions.y {
            for x in 0..dimensions.x {
                let z = (dimensions.z+1)/2;
                let ix = x*dimensions.z*dimensions.y+y*dimensions.z+z;
                let byte_ix = (ix*4) as usize;
                let float = f32::from_le_bytes(sdf.data[byte_ix..byte_ix+4].try_into().unwrap());
                let int = (float - 0.5).ceil() as i32;
                if int < 0 {
                    // let point = aabb.min() + scale * ((UVec3::new(x as u32,y as u32,z as u32) * 2) + 1).as_vec3a();
                    // print!("{}:{}  ", point, float);
                    print!(" {}", int);
                } else {
                    print!("  {}", int);
                }
            }
            println!("");
        }
        assert!(false);
    }

    #[test]
    fn sphere_sdf() {
        let func = SdfFunc::sphere(Vec3::ZERO, 6.5);
        let aabb = Aabb{center: Vec3A::ZERO, half_extents: Vec3A::splat(10.0) };
        let dimensions = UVec3::new(25, 25, 25);
        let sdf = create_sdf_from_fn(&func, &aabb, dimensions);

        for y in 0..dimensions.y {
            for x in 0..dimensions.x {
                let z = (dimensions.z+1)/2;
                let ix = x*dimensions.z*dimensions.y+y*dimensions.z+z;
                let byte_ix = (ix*4) as usize;
                let float = f32::from_le_bytes(sdf.data[byte_ix..byte_ix+4].try_into().unwrap());
                let int = (float - 0.5).ceil() as i32;
                if int < 0 {
                    // let point = aabb.min() + scale * ((UVec3::new(x as u32,y as u32,z as u32) * 2) + 1).as_vec3a();
                    // print!("{}:{}  ", point, float);
                    print!(" {}", int);
                } else {
                    print!("  {}", int);
                }
            }
            println!("");
        }
        assert!(false);
    }

    #[test]
    fn tri_sdf() {
        let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]);
        let aabb = Aabb{ center: Vec3A::ZERO, half_extents: Vec3A::splat(3.0) };
        let dimensions = UVec3::new(5, 5, 5);
        let sdf = create_sdf_from_mesh_cpu(&mesh, &aabb, dimensions);
        println!("{:?}", aabb);
        println!("{:?}", sdf.data);

        // let scale = aabb.half_extents / dimensions.as_vec3a();

        for y in 0..dimensions.y {
            for x in 0..dimensions.x {
                let z = (dimensions.z+1)/2;
                let ix = x*dimensions.z*dimensions.y+y*dimensions.z+z;
                let byte_ix = (ix*4) as usize;
                let float = f32::from_le_bytes(sdf.data[byte_ix..byte_ix+4].try_into().unwrap());
                if float <= 0.0 {
                    // let point = aabb.min() + scale * ((UVec3::new(x as u32,y as u32,z as u32) * 2) + 1).as_vec3a();
                    // print!("{}:{}  ", point, float);
                    print!("x");
                } else {
                    print!(" ");
                }
            }
            println!("");
        }
    }

    #[test]
    fn cube_sdf() {
        let mesh: Mesh = shape::Box::new(1.0, 1.0, 1.0).into();
        println!("triangles: {}", mesh.indices().unwrap().len() / 3);
        let aabb = Aabb{ center: Vec3A::ZERO, half_extents: Vec3A::splat(1.0) };
        let dimensions = UVec3::new(24, 24, 24);
        let sdf = create_sdf_from_mesh_cpu(&mesh, &aabb, dimensions);
        println!("{:?}", aabb);
        // println!("{:?}", sdf.data);

        // let scale = aabb.half_extents / dimensions.as_vec3a();

        for y in 0..dimensions.y {
            for x in 0..dimensions.x {
                let z = (dimensions.z+1)/2;
                let ix = x*dimensions.z*dimensions.y+y*dimensions.z+z;
                let byte_ix = (ix*4) as usize;
                let float = f32::from_le_bytes(sdf.data[byte_ix..byte_ix+4].try_into().unwrap());
                if float <= 0.0 {
                    // let point = aabb.min() + scale * ((UVec3::new(x as u32,y as u32,z as u32) * 2) + 1).as_vec3a();
                    // print!("{}:{}  ", point, float);
                    print!("x");
                } else {
                    print!(".");
                }
            }
            println!("");
        }
        assert!(false);
    }
}