#![feature(let_else, slice_as_chunks, bool_to_option)]
pub mod controller;
pub mod render;
pub mod shader;

use std::{collections::BTreeMap};

use bevy::{
    math::Vec3A,
    prelude::*,
    render::{
        mesh::{PrimitiveTopology, VertexAttributeValues},
        primitives::{Aabb, Plane},
        render_resource::{Extent3d, TextureDimension, SamplerDescriptor, AddressMode, FilterMode}, texture::ImageSampler,
    }, utils::FloatOrd,
};

pub struct Mesh2Sdf;

impl Plugin for Mesh2Sdf {
    fn build(&self, app: &mut App) {
        app.add_system(generate_sdfs);
    }
}

#[derive(Component)]
pub struct Sdf {
    pub image: Handle<Image>,
    pub aabb: Aabb,
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

    pub fn union(funcs: impl Iterator<Item = SdfFunc>) -> SdfFunc {
        let funcs: Vec<_> = funcs.collect();
        Self(Box::new(move |p: Vec3| {
            funcs.iter().fold(f32::MAX, |min, f| f32::min(min, f.0(p)))
        }))
    }
}

#[derive(Debug)]
pub struct TriData {
    pub a: Vec3A,
    pub b: Vec3A,
    pub c: Vec3A,
}

pub fn create_sdf_from_mesh_gpu(mesh: &Mesh, aabb: &Aabb, dimension: UVec3) -> Image {
    assert!(
        matches!(mesh.primitive_topology(), PrimitiveTopology::TriangleList),
        "`sdf generation can only work on `TriangleList`s"
    );

    let Some(VertexAttributeValues::Float32x3(values)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        panic!("bad mesh");
    };

    let values: Vec<[f32; 3]> = match mesh.indices() {
        Some(ix) => ix.iter().map(|ix| values[ix]).collect(),
        None => values.iter().copied().collect(),
    };

    let scale = aabb.half_extents / dimension.as_vec3a();
    let shift = scale * 0.2;

    let _triangles: Vec<TriData> = values
        .chunks_exact(3)
        .map(|abc| {
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
        })
        .collect();

    Image::new_fill(
        Extent3d {
            width: dimension.x,
            height: dimension.y,
            depth_or_array_layers: dimension.z,
        },
        TextureDimension::D3,
        &f32::MAX.to_le_bytes(),
        bevy::render::render_resource::TextureFormat::R32Float,
    )
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct OrderedVec(Vec3A);

impl Eq for OrderedVec {}
impl Ord for OrderedVec {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        FloatOrd(self.0.x).cmp(&FloatOrd(other.0.x))
            .then(FloatOrd(self.0.y).cmp(&FloatOrd(other.0.y)))
            .then(FloatOrd(self.0.z).cmp(&FloatOrd(other.0.z)))
    }
}

impl PartialOrd for OrderedVec {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&other))
    }
}

pub fn create_sdf_from_mesh_cpu(mesh: &Mesh, aabb: &Aabb, dimension: UVec3, debug: Option<UVec3>) -> Image {
    let start = std::time::Instant::now();
    assert!(
        matches!(mesh.primitive_topology(), PrimitiveTopology::TriangleList),
        "`sdf generation can only work on `TriangleList`s"
    );

    let Some(VertexAttributeValues::Float32x3(values)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
        panic!("bad mesh");
    };

    let values: Vec<[f32; 3]> = match mesh.indices() {
        Some(ix) => ix.iter().map(|ix| values[ix]).collect(),
        None => values.iter().copied().collect(),
    };

    let scale = aabb.half_extents * 2.0 / (dimension - 1).as_vec3a();

    #[derive(Debug)]
    struct TriData {
        a: Vec3A,
        b: Vec3A,
        c: Vec3A,
        inv_area: f32,
        plane: Plane,
    }

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
        let (a, b, c) = (iter.next().unwrap(), iter.next().unwrap(), iter.next().unwrap());

        let ab_len = (b.0 - a.0).length();
        let ac_len = (c.0 - a.0).length();
        let bc_len = (c.0 - b.0).length();

        let a_angle = tri_angle(bc_len, ab_len, ac_len);
        let b_angle = tri_angle(ac_len, ab_len, bc_len);
        let c_angle = tri_angle(ab_len, ac_len, bc_len);

        *vertices.entry(a).or_default() += normal * a_angle;
        *vertices.entry(b).or_default() += normal * b_angle;
        *vertices.entry(c).or_default() += normal * c_angle;

        *edges.entry((a,b)).or_default() += normal;
        *edges.entry((a,c)).or_default() += normal;
        *edges.entry((b,c)).or_default() += normal;

        let plane = Plane::new(normal.extend(-(a.0).dot(normal)));
        let inv_area = (b.0 - a.0).cross(c.0 - a.0).dot(plane.normal()).recip();

        triangles.push(TriData { a: a.0, b: b.0, c: c.0, inv_area, plane });
    }

    fn tri_angle(opp: f32, a: f32, b: f32) -> f32 {
        ((a*a + b*b - opp*opp) / (2.0 * a * b)).acos()
    }

    let compute_distance = |point: Vec3A, debug: bool| -> f32 {
        if debug {
            println!("point: {}", point);
        }

        #[derive(Default, Debug)]
        struct Res {
            dist_sq: f32,
            norm: Vec3A,
            nearest: Vec3A,
        }

        let mut best = Res{dist_sq: f32::MAX, ..Default::default()};

        for (v, n) in vertices.iter() {
            let dist_sq = point.distance_squared(v.0);
            if dist_sq < best.dist_sq {
                best.dist_sq = dist_sq;
                best.norm = *n;
                best.nearest = v.0;
                if debug {println!("vertex -- {}\n{:?}", v.0, best);}
            }
        }

        for ((v0, v1), n) in edges.iter() {
            let line = v1.0 - v0.0;
            let line_len_sq = line.length_squared();
            let intercept = f32::clamp((point - v0.0).dot(line), 0.0, line_len_sq);
            if intercept < 0.001 || intercept > line_len_sq * 0.999 {
                continue;
            }

            let nearest = v0.0 + line * (intercept / line_len_sq);
            let dist_sq = point.distance_squared(nearest);
            if dist_sq < best.dist_sq {
                best.dist_sq = dist_sq;
                best.norm = *n;
                best.nearest = nearest;
                if debug {println!("edge -- {}-{}\n{:?}", v0.0, v1.0, best);}
            }
        }

        for tri in triangles.iter() {
            let distance_to_plane = tri.plane.normal_d().dot(point.extend(1.0));
            let distance_to_plane_sq = distance_to_plane * distance_to_plane;
            if distance_to_plane_sq > best.dist_sq {
                continue;
            }

            let point_on_plane = point - distance_to_plane * tri.plane.normal();
            // barycentric coords
            let u = (tri.c - tri.b)
                .cross(point_on_plane - tri.b)
                .dot(tri.plane.normal())
                * tri.inv_area;
            let v = (tri.a - tri.c)
                .cross(point_on_plane - tri.c)
                .dot(tri.plane.normal())
                * tri.inv_area;
            let w = 1.0 - u - v;

            if u.is_sign_positive() && v.is_sign_positive() && w.is_sign_positive() {
                best.dist_sq = distance_to_plane_sq;
                best.norm = tri.plane.normal();
                best.nearest = point_on_plane;
                if debug {println!("tri -- {:?}\n{:?}", tri, best);}
            }
        }

        let direction = point - best.nearest;
        let outside = direction.dot(best.norm) >= 0.0;

        if debug {println!("dist {}", best.dist_sq.sqrt() * direction.dot(best.norm).signum());}

        if outside {
            best.dist_sq.sqrt()
        } else {
            -best.dist_sq.sqrt()
        }
    };

    let mut data: Vec<u8> = Vec::new();
    data.resize((4 * dimension.x * dimension.y * dimension.z) as usize, 0);
    let mut chunks = data.as_chunks_mut::<4>().0.iter_mut();

    let prep = std::time::Instant::now();

    for z in 0..dimension.z {
        for y in 0..dimension.y {
            for x in 0..dimension.x {
                let point = aabb.min() + scale * UVec3::new(x, y, z).as_vec3a();

                if Some(UVec3::new(x, y, z)) == debug {
                    compute_distance(point, true);
                }

                let dist = compute_distance(point, false);

                // if (dist + 0.6664815).abs() < 0.01 {
                //     println!("found you: {:?}", [x, y, z]);
                // }

                let chunk = chunks.next().unwrap();
                chunk.copy_from_slice(&dist.to_le_bytes());
            }
        }
    }

    let process = std::time::Instant::now();

    let mut image = Image::new(
        Extent3d {
            width: dimension.x,
            height: dimension.y,
            depth_or_array_layers: dimension.z,
        },
        TextureDimension::D3,
        data,
        bevy::render::render_resource::TextureFormat::R32Float,
    );

    image.sampler_descriptor = ImageSampler::Descriptor(SamplerDescriptor{
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..Default::default()
    });

    let res = std::time::Instant::now();

    println!("prep: {:?}, proc: {:?}, res: {:?}, tot: {:?}", prep - start, process - prep, res - process, res - start);

    image
}

fn create_sdf_from_fn(func: &SdfFunc, aabb: &Aabb, dimension: UVec3) -> Image {
    let scale = Vec3::from(aabb.half_extents) / dimension.as_vec3();

    let mut data: Vec<u8> = Vec::new();
    data.resize((4 * dimension.x * dimension.y * dimension.z) as usize, 0);
    let mut chunks = data.as_chunks_mut::<4>().0.iter_mut();

    for x in 0..dimension.x {
        for y in 0..dimension.y {
            for z in 0..dimension.z {
                let point =
                    Vec3::from(aabb.min()) + scale * ((UVec3::new(x, y, z) * 2) + 1).as_vec3();
                let dist = func.0(point);
                let chunk = chunks.next().unwrap();
                chunk.copy_from_slice(&dist.to_le_bytes());
            }
        }
    }

    Image::new(
        Extent3d {
            width: dimension.x,
            height: dimension.y,
            depth_or_array_layers: dimension.z,
        },
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
            let image = create_sdf_from_mesh_cpu(mesh, aabb, autosdf.dimensions, None);
            let image = images.add(image);
            commands.entity(ent).insert(Sdf {
                image,
                aabb: aabb.clone(),
            });
        }
    }

    for (ent, aabb, anal) in new_by_anal.iter() {
        let image = create_sdf_from_fn(&anal.func, aabb, anal.dimension);
        let image = images.add(image);
        commands.entity(ent).insert(Sdf {
            image,
            aabb: aabb.clone(),
        });
    }
}

#[cfg(test)]
mod test {
    use bevy::{
        math::Vec3A,
        prelude::*,
        render::{mesh::PrimitiveTopology, primitives::Aabb},
    };

    use crate::{create_sdf_from_fn, create_sdf_from_mesh_cpu, SdfFunc};

    #[test]
    fn icosphere_sdf() {
        let icosphere = shape::Icosphere {
            radius: 6.5,
            subdivisions: 10,
        };
        let mesh = Mesh::from(icosphere);
        println!("vertices: {}", mesh.count_vertices());
        let aabb = Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(10.0),
        };
        let dimensions = UVec3::new(25, 25, 25);
        let sdf = create_sdf_from_mesh_cpu(&mesh, &aabb, dimensions, None);

        let _scale = aabb.half_extents / dimensions.as_vec3a();

        for y in 0..dimensions.y {
            for x in 0..dimensions.x {
                let z = (dimensions.z + 1) / 2;
                let ix = x * dimensions.z * dimensions.y + y * dimensions.z + z;
                let byte_ix = (ix * 4) as usize;
                let float = f32::from_le_bytes(sdf.data[byte_ix..byte_ix + 4].try_into().unwrap());
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
        let aabb = Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(10.0),
        };
        let dimensions = UVec3::new(25, 25, 25);
        let sdf = create_sdf_from_fn(&func, &aabb, dimensions);

        for y in 0..dimensions.y {
            for x in 0..dimensions.x {
                let z = (dimensions.z + 1) / 2;
                let ix = x * dimensions.z * dimensions.y + y * dimensions.z + z;
                let byte_ix = (ix * 4) as usize;
                let float = f32::from_le_bytes(sdf.data[byte_ix..byte_ix + 4].try_into().unwrap());
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
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        );
        let aabb = Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(3.0),
        };
        let dimensions = UVec3::new(5, 5, 5);
        let sdf = create_sdf_from_mesh_cpu(&mesh, &aabb, dimensions, None);
        println!("{:?}", aabb);
        println!("{:?}", sdf.data);

        // let scale = aabb.half_extents / dimensions.as_vec3a();

        for y in 0..dimensions.y {
            for x in 0..dimensions.x {
                let z = (dimensions.z + 1) / 2;
                let ix = x * dimensions.z * dimensions.y + y * dimensions.z + z;
                let byte_ix = (ix * 4) as usize;
                let float = f32::from_le_bytes(sdf.data[byte_ix..byte_ix + 4].try_into().unwrap());
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
        let aabb = Aabb {
            center: Vec3A::ZERO,
            half_extents: Vec3A::splat(1.0),
        };
        let dimensions = UVec3::new(24, 24, 24);
        let sdf = create_sdf_from_mesh_cpu(&mesh, &aabb, dimensions, None);
        println!("{:?}", aabb);
        // println!("{:?}", sdf.data);

        // let scale = aabb.half_extents / dimensions.as_vec3a();

        for y in 0..dimensions.y {
            for x in 0..dimensions.x {
                let z = (dimensions.z + 1) / 2;
                let ix = x * dimensions.z * dimensions.y + y * dimensions.z + z;
                let byte_ix = (ix * 4) as usize;
                let float = f32::from_le_bytes(sdf.data[byte_ix..byte_ix + 4].try_into().unwrap());
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
