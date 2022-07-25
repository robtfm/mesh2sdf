struct VertexData {
    v: vec3<f32>,
    n: vec3<f32>,
};

struct Vertices {
    data: array<VertexData>,
};

struct EdgeData {
    a: vec3<f32>,
    b: vec3<f32>,
    n: vec3<f32>,
};

struct Edges {
    data: array<EdgeData>,
};

struct TriData {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    plane: vec4<f32>,
    inv_area: f32,
};

struct Tris {
    data: array<TriData>,
};

struct InstanceData {
    write_position: vec3<u32>,
    aabb_min: vec3<f32>,
    scale: vec3<f32>,
    block_dimensions: vec3<u32>,
    counts: vec3<u32>,
    block_count: u32,
};

struct Instances {
    data: array<InstanceData>,
};

@group(0) @binding(0)
var<storage> instances: Instances;
@group(0) @binding(1)
var<storage> vertices: Vertices;
@group(0) @binding(2)
var<storage> edges: Edges;
@group(0) @binding(3)
var<storage> tris: Tris;
@group(0) @binding(4)
var texture: texture_storage_3d<r32float, write>;

fn distance_squared(x: vec3<f32>, y: vec3<f32>) -> f32 {
    let v = y - x;
    return dot(v, v);
}

@compute 
@workgroup_size(8, 8, 8)
fn calc(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    var block_id = invocation_id.x / 8u;
    var instance_index = 0;

    var start = vec3<u32>(0u, 0u, 0u);
    var instance = instances.data[instance_index];
    while (block_id >= instance.block_count) {
        start = start + instance.counts;
        block_id = block_id - instance.block_count;
        instance_index = instance_index + 1;
        instance = instances.data[instance_index];
    }

    let block_z = block_id / (instance.block_dimensions.x * instance.block_dimensions.y);
    let block_y = (block_id - block_z * (instance.block_dimensions.x * instance.block_dimensions.y)) / instance.block_dimensions.x;
    let block_x = (block_id - block_z * (instance.block_dimensions.x * instance.block_dimensions.y) - block_y * instance.block_dimensions.x);

    let target_offset = vec3<u32>(
        block_x * 8u + invocation_id.x % 8u,
        block_y * 8u + invocation_id.y,
        block_z * 8u + invocation_id.z,
    );

    let target_point: vec3<f32> = instance.aabb_min + vec3<f32>(target_offset) * instance.scale;

    var best_dist_sq = 999999.0;
    var best_norm: vec3<f32>;
    var best_nearest: vec3<f32>;

    for (var i = start.x; i < start.x + instance.counts.x; i = i + 1u) {
        let data = vertices.data[i];
        let dist_sq = distance_squared(target_point, data.v);
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_norm = data.n;
            best_nearest = data.v;
        }
    }

    for (var i = start.y; i < start.y + instance.counts.y; i = i + 1u) {
        let data = edges.data[i];

        let edge = data.b - data.a;
        let edge_len_sq = dot(edge, edge);
        let intercept = clamp(dot(target_point - data.a, edge), 0.0, edge_len_sq);
        if (intercept < 0.001 || intercept > edge_len_sq * 0.999) {
            continue;
        }

        let nearest = data.a + edge * (intercept / edge_len_sq);
        let dist_sq = distance_squared(target_point, nearest);
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_norm = data.n;
            best_nearest = nearest;
        }
    }

    for (var i = start.z; i < start.z + instance.counts.z; i = i + 1u) {
        let tri = tris.data[i];

        let distance_to_plane = dot(tri.plane, vec4<f32>(target_point, 1.0));
        let distance_to_plane_sq = distance_to_plane * distance_to_plane;
        if (distance_to_plane_sq > best_dist_sq) {
            continue;
        }

        let n = tri.plane.xyz;
        let point_on_plane = target_point - distance_to_plane * n;
        // barycentric coords
        let u = dot(
                    cross(tri.c - tri.b, point_on_plane - tri.b),
                    n
                ) * tri.inv_area;
        let v = dot(
                    cross(tri.a - tri.c, point_on_plane - tri.c),
                    n
                ) * tri.inv_area;
        let w = 1.0 - u - v;

        if (u >= 0.0 && v >= 0.0 && w >= 0.0) {
            best_dist_sq = distance_to_plane_sq;
            best_norm = tri.plane.xyz;
            best_nearest = point_on_plane;
        }
    }

    let direction = target_point - best_nearest;
    let outside = sign(dot(direction, best_norm));
    let dist = sqrt(best_dist_sq) * outside;

    textureStore(texture, vec3<i32>(instance.write_position + target_offset), vec4<f32>(dist, 0.0, 0.0, 1.0));
}