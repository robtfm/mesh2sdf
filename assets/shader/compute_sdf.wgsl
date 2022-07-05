struct VertexData {
    v: vec3<f32>;
    n: vec3<f32>;
};

struct Vertices {
    data: array<VertexData>;
};

struct EdgeData {
    a: vec3<f32>;
    b: vec3<f32>;
    n: vec3<f32>;
};

struct Edges {
    data: array<EdgeData>;
};

struct TriData {
    a: vec3<f32>;
    b: vec3<f32>;
    c: vec3<f32>;
    plane: vec4<f32>;
    inv_area: f32;
};

struct Tris {
    data: array<TriData>;
};

struct Sdf {
    aabb_min: vec3<f32>;
    scale: vec3<f32>;
};

[[group(0), binding(0)]]
var<uniform> sdf: Sdf;
[[group(0), binding(1)]]
var<storage> vertices: Vertices;
[[group(0), binding(2)]]
var<storage> edges: Edges;
[[group(0), binding(3)]]
var<storage> tris: Tris;
[[group(0), binding(4)]]
var texture: texture_storage_3d<r32float, write>;

fn distance_squared(x: vec3<f32>, y: vec3<f32>) -> f32 {
    let v = y - x;
    return dot(v, v);
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn calc([[builtin(global_invocation_id)]] invocation_id: vec3<u32>) {
    var best_dist_sq = 999999.0;
    var best_norm: vec3<f32>;
    var best_nearest: vec3<f32>;

    let point: vec3<f32> = sdf.aabb_min + vec3<f32>(invocation_id) * sdf.scale;

    for (var i = 0u; i < arrayLength(&vertices.data); i = i + 1u) {
        let data = vertices.data[i];
        let dist_sq = distance_squared(point, data.v);
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_norm = data.n;
            best_nearest = data.v;
        }
    }

    for (var i = 0u; i < arrayLength(&edges.data); i = i + 1u) {
        let data = edges.data[i];

        let line = data.b - data.a;
        let line_len_sq = dot(line, line);
        let intercept = clamp(dot(point - data.a, line), 0.0, line_len_sq);
        if (intercept < 0.001 || intercept > line_len_sq * 0.999) {
            continue;
        }

        let nearest = data.a + line * (intercept / line_len_sq);
        let dist_sq = distance_squared(point, nearest);
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_norm = data.n;
            best_nearest = nearest;
        }
    }

    for (var i = 0u; i < arrayLength(&tris.data); i = i + 1u) {
        let tri = tris.data[i];

        let distance_to_plane = dot(tri.plane, vec4<f32>(point, 1.0));
        let distance_to_plane_sq = distance_to_plane * distance_to_plane;
        if (distance_to_plane_sq > best_dist_sq) {
            continue;
        }

        let n = tri.plane.xyz;
        let point_on_plane = point - distance_to_plane * n;
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

    let direction = point - best_nearest;
    let outside = sign(dot(direction, best_norm));
    let dist = sqrt(best_dist_sq) * outside;
    textureStore(texture, vec3<i32>(invocation_id), vec4<f32>(dist, 0.0, 0.0, 1.0));
}