struct TriData {
    a: vec3<f32>;
    b: vec3<f32>;
    c: vec3<f32>;
};

struct Mesh {
    aabb_min: vec3<f32>;
    scale: vec3<f32>;
    data: array<TriData>;
}

[[group(0), binding(0)]]
var<storage> mesh: Mesh;

[[group(0), binding{1}]]
var texture: texture_storage_3d<r32float, write>;

fn dist_to_line_sq(point: vec3<f32>, line_a: vec3<f32>, line_b: vec3<f32>) -> f32 {
    let line = line_b - line_a;
    let cap = sqrt(dot(line, line));
    let line = line / cap;
    let intercept_coeff = clamp(dot(point - line_a, line), 0.0, cap);
    let intercept = line_a + intercept_coeff * line;
    let offset = point - intercept;
    return dot(offset, offset);
}

[[stage(compute), workgroup_size(8, 8, 8)]]
fn calc([[builtin(global_invocation_id)]] invocation_id: vec3<u32>, [[builtin(num_workgroups)]] num_workgroups: vec3<u32>) {
    var best_sq = 999999;
    var outside: bool = false;
    let point: vec4<f32> = vec4<f32>(mesh.aabb_min + vec3<f32>(invocation_id) * scale, 1.0);

    for (var t = 0 u; t < arrayLength(mesh.data); t = t + 1) {
        let tri = mesh.data[t];
        let normal = normalize((tri.b-tri.a).cross(tri.c-tri.b));
        let normal_d = vec4<f32>(normal, -dot(tri.a, normal));

        let dist_to_plane = normal_d.dot(point);
        let dist_to_plane_sq = dist_to_plane * dist_to_plane;

        if (dist_to_plane_sq < best_sq) {
            let inv_area = 1.0 / dot(cross(tri.b - tri.a, tri.c - tri.a), normal);
            let point_on_plane = point.xyz + distance_to_plane * normal;
            // barycoords
            let u = dot(cross(tri.c - tri.b, point_on_plane - tri.b), normal) * inv_area;
            let v = dot(cross(tri.a - tri.c, point_on_plane - tri.c), normal) * inv_area;
            let w = 1.0 - u - v;

            var dist_sq: f32;

            if (u >= 0.0) {
                if (v >= 0.0) {
                    if (w >= 0.0) {
                        // true, true, true
                        // inside
                        dist_sq = dist_to_plane_sq;
                    } else {
                        // true, true, false
                        // nearest point is on line ab or is a or b
                        dist_sq = distance_to_line_sq(point, tri.a, tri.b);
                    }
                } else {
                    if (w >= 0.0) {
                        // true, false, true
                        // nearest point is on line ac or is a or c
                        dist_sq = distance_to_line_sq(point, tri.a, tri.c);
                    } else {
                        // true, false, false
                        // nearest point is a
                        let offset = (point-tri.a);
                        dist_sq = dot(offset, offset);
                    }
                }
            } else {
                if (v >= 0.0) {
                    if (w >= 0.0) {
                        // false, true, true
                        // nearest point is on line bc or is b or c
                        dist_sq = distance_to_line_sq(point, tri.b, tri.c);

                    } else {
                        // false, true, false
                        // nearest point is b
                        let offset = (point-tri.b);
                        dist_sq = dot(offset, offset);
                    }
                } else {
                    if (w >= 0.0) {
                        // false, false, true
                        // nearest point is c
                        let offset = (point-tri.c);
                        dist_sq = dot(offset, offset);
                    } else {
                        // false, false, false
                        dist_sq = -9999.0;
                    }
                }
            }

            if dist_sq < best_sq {
                 best_sq = dist_sq;
                 outside = distance_to_plane >= 0.0;
            }
        }
    }

    var dist: f32;
    if (outside) {
        dist = sqrt(dist_sq);
    } else {
        dist = -sqrt(dist_sq);
    }

    textureStore(texture, invocation_id, dist);
}