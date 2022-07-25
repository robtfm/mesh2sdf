use bevy::{
    math::Vec3A,
    prelude::*,
    render::{
        mesh::PrimitiveTopology,
        primitives::Aabb,
        render_resource::{AddressMode, Extent3d, FilterMode, SamplerDescriptor, TextureDimension},
        texture::ImageSampler,
    },
};

use crate::utils::preprocess_mesh_for_sdf;

pub fn create_sdf_from_mesh_cpu(
    mesh: &Mesh,
    aabb: &Aabb,
    dimension: UVec3,
    debug: Option<UVec3>,
) -> Image {
    let start = std::time::Instant::now();
    assert!(
        matches!(mesh.primitive_topology(), PrimitiveTopology::TriangleList),
        "`sdf generation can only work on `TriangleList`s"
    );

    let preprocessed = preprocess_mesh_for_sdf(mesh, None);

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

        let mut best = Res {
            dist_sq: f32::MAX,
            ..Default::default()
        };

        for &(v, n) in preprocessed.vertices.iter() {
            let dist_sq = point.distance_squared(v);
            if dist_sq < best.dist_sq {
                best.dist_sq = dist_sq;
                best.norm = n;
                best.nearest = v;
                if debug {
                    println!("vertex -- {}\n{:?}", v, best);
                }
            }
        }

        for &((v0, v1), n) in preprocessed.edges.iter() {
            let line = v1 - v0;
            let line_len_sq = line.length_squared();
            let intercept = f32::clamp((point - v0).dot(line), 0.0, line_len_sq);
            if intercept < 0.001 || intercept > line_len_sq * 0.999 {
                continue;
            }

            let nearest = v0 + line * (intercept / line_len_sq);
            let dist_sq = point.distance_squared(nearest);
            if dist_sq < best.dist_sq {
                best.dist_sq = dist_sq;
                best.norm = n;
                best.nearest = nearest;
                if debug {
                    println!("edge -- {}-{}\n{:?}", v0, v1, best);
                }
            }
        }

        for tri in preprocessed.triangles.iter() {
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
                if debug {
                    println!("tri -- {:?}\n{:?}", tri, best);
                }
            }
        }

        let direction = point - best.nearest;
        let outside = direction.dot(best.norm) >= 0.0;

        if debug {
            println!(
                "dist {}",
                best.dist_sq.sqrt() * direction.dot(best.norm).signum()
            );
        }

        if outside {
            best.dist_sq.sqrt()
        } else {
            -best.dist_sq.sqrt()
        }
    };

    let scale = aabb.half_extents * 2.0 / (dimension - 1).as_vec3a();

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

    image.sampler_descriptor = ImageSampler::Descriptor(SamplerDescriptor {
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..Default::default()
    });

    let res = std::time::Instant::now();

    println!(
        "prep: {:?}, proc: {:?}, res: {:?}, tot: {:?}",
        prep - start,
        process - prep,
        res - process,
        res - start
    );

    image
}
