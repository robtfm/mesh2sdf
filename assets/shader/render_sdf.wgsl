#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::mesh_bindings
#import bevy_pbr::mesh_functions

struct Uniform {
    aabb_min: vec3<f32>;
    aabb_extents: vec3<f32>;
    base_color: vec4<f32>;
    hit_color: vec4<f32>;
    step_color: vec4<f32>;
    distance_color: vec4<f32>;
    min_step_size: f32;
    hit_threshold: f32;
    max_step_count: u32;
};

[[group(1), binding(0)]]
var<uniform> material: Uniform;
[[group(1), binding(1)]]
var sdf_texture: texture_3d<f32>;
[[group(1), binding(2)]]
var sdf_sampler: sampler;

struct FragmentInput {
    [[builtin(position)]] frag_coord: vec4<f32>;
};

fn isnan(x: f32) -> bool {
    if (x == 0.0) {return false;}
    if (min(0.0, x) == max(0.0, x)) {return true;}
    return false;
}

fn sample_distance(pos: vec3<f32>) -> vec3<f32> {
    let local_position = transpose(mesh.inverse_transpose_model) * vec4<f32>(pos, 1.0);
    let local_position = local_position.xyz / local_position.w;
    let nearest = clamp(local_position, material.aabb_min, material.aabb_min + material.aabb_extents);

    let coords = clamp((local_position - material.aabb_min) / material.aabb_extents, vec3<f32>(0.0), vec3<f32>(1.0)); // 0-1
    let sample = textureSample(sdf_texture, sdf_sampler, coords).r;

    let offset = nearest - local_position;
    let distance_to_aabb_sq = dot(offset, offset);        

    if (distance_to_aabb_sq == 0.0) {
        // inside the volume
        return vec3<f32>(sample, 0.0, 0.0);
    } else {
        if (sample < 0.0) {
            return vec3<f32>(0.0, 1.0, 0.0);
            // error, negative on edge
        }
        // worst case assume right angle
        let sample = max(sample, 0.0);
        return vec3<f32>(sqrt(distance_to_aabb_sq + sample * sample), 0.0, 0.0);
    }
}

[[stage(fragment)]]
fn fragment(in: FragmentInput) -> [[location(0)]] vec4<f32> {
    let x = in.frag_coord.x / f32(view.width) * 2.0 - 1.0;
    let y = in.frag_coord.y / f32(view.height) * -2.0 + 1.0;

    let origin = view.inverse_view_proj * vec4<f32>(x, y, 1.0, 1.0);
    let origin = origin.xyz / origin.w;
    let offset = view.inverse_view_proj * vec4<f32>(x, y, 0.5, 1.0);
    let offset = offset.xyz / offset.w;

    let ray = normalize(offset - origin);
    let distance_to_origin = sqrt(dot(origin, origin));

    var pos = origin;
    var steps: u32 = 0u;

    var dist = material.hit_threshold + 0.001;
    var distance_sq = 0.0;

    // todo use view depth for max dist
    // also use aabb center + half_extents?
    let tl = mesh.model * vec4<f32>(material.aabb_min, 1.0);
    let tl = tl.xyz / tl.w;
    let max_distance = distance(origin, tl) + length(material.aabb_extents) * 2.0;
    let max_distance_sq = max_distance * max_distance;

    for (; distance_sq < max_distance_sq && dist > material.hit_threshold && steps < material.max_step_count; steps = steps + 1u) {
        pos = pos + max(material.min_step_size, dist) * ray;
        let res = sample_distance(pos);
        if (res.y > 0.0) {
            return vec4<f32>(0.0, sqrt(res.x), res.z, 1.0);
        }
        dist = res.x;
        distance_sq = dot(pos - origin, pos - origin);
    }

    var output = material.base_color;
    if (dist < material.hit_threshold) {
        output = output + material.hit_color;
    }
    output = output + material.step_color * (f32(steps) / f32(material.max_step_count));
    output = output + sqrt(distance_sq / max_distance_sq) * material.distance_color;

    return output;
}