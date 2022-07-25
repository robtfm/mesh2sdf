struct SdfUniform {
    // tap distances
    ao_distances: vec3<f32>,
    // cone angle (opp/adj)
    ao_sin_angle: f32,
};

struct SdfHeader {
    transform: mat4x4<f32>,
    aabb_min: vec3<f32>,
    aabb_size: vec3<f32>,
    atlas_position: vec3<f32>,
    atlas_size: vec3<f32>,
    scale: f32,
};

struct SdfHeaders {
    data: array<SdfHeader>,
};

struct SdfClusterIndexes {
    // vec4<u32>(offset, count, 0, 0)
    data: array<vec2<u32>>,
};

@group(0) @binding(0)
var<uniform> sdf_view: SdfUniform;

@group(0) @binding(1)
var<storage> sdf_headers: SdfHeaders;

// @group(0) @binding(2)
// var<storage> sdf_clusters: SdfClusterIndexes;

@group(0) @binding(2)
var sdf_atlas: texture_3d<f32>;
@group(0) @binding(3)
var sdf_sampler: sampler;
