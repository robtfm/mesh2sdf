#define_import_path bevy_pbr::pbr_ambient

fn sdf_item_distance(target_point: vec3<f32>, index: u32) -> f32 {
    let sdf_header = sdf_headers.data[index];

    let local_position = sdf_header.transform * vec4<f32>(target_point, 1.0);
    let local_position = local_position.xyz / local_position.w;
    let nearest = clamp(local_position, sdf_header.aabb_min.xyz, sdf_header.aabb_min.xyz + sdf_header.aabb_size.xyz);

    let coords = clamp((local_position - sdf_header.aabb_min.xyz) / sdf_header.aabb_size.xyz, vec3<f32>(0.0), vec3<f32>(1.0)); // 0-1
    let atlas_coords = sdf_header.atlas_position.xyz + coords * sdf_header.atlas_size.xyz;
    let inner_distance = textureSample(sdf_atlas, sdf_sampler, atlas_coords).r;

    let offset = nearest - local_position;
    let distance_to_aabb_sq = dot(offset, offset);        

    if (distance_to_aabb_sq == 0.0) {
        // inside the volume
        // return vec3<f32>(inner_distance, 0.0, 0.0);
        return inner_distance * sdf_header.scale;
    } else {
        // if (inner_distance < 0.0) {
        //     return vec3<f32>(0.0, 1.0, 0.0);
        //     // error, negative on edge
        // }
        // worst case assume right angle
        // let inner_distance = max(inner_distance, 0.0);
        // return vec3<f32>(sqrt(distance_to_aabb_sq + inner_distance * inner_distance), 0.0, 0.0);
        // return sqrt(distance_to_aabb_sq + inner_distance * inner_distance);
        return 999.0;
    }
}

fn sdf_distance(target_point: vec3<f32>, max_distance: f32) -> f32 {
    var distance = max_distance;

    for (var i = 0u; i < arrayLength(&sdf_headers.data); i = i + 1u) {
        let item_distance = sdf_item_distance(target_point, i);
        distance = min(item_distance, distance);
    }
    return distance;
}

fn sdf_occlusion(world_position: vec4<f32>, world_normal: vec3<f32>, cone_scale: f32) -> f32 {
    let target_point = world_position.xyz + world_normal * sdf_view.ao_distances.x;
    let cone_radius = sdf_view.ao_distances.x * sdf_view.ao_sin_angle * cone_scale;
    let target_point_distance = sdf_distance(target_point, cone_radius) / cone_radius;
    let close = 1.0 - clamp(target_point_distance, 0.0, 1.0);

    let target_point = world_position.xyz + world_normal * sdf_view.ao_distances.y;
    let cone_radius = sdf_view.ao_distances.y * sdf_view.ao_sin_angle * cone_scale;
    let target_point_distance = sdf_distance(target_point, cone_radius) / cone_radius;
    let mid = 1.0 - clamp(target_point_distance, 0.0, 1.0);

    let target_point = world_position.xyz + world_normal * sdf_view.ao_distances.z;
    let cone_radius = sdf_view.ao_distances.z * sdf_view.ao_sin_angle * cone_scale;
    let target_point_distance = sdf_distance(target_point, cone_radius) / cone_radius;
    let far = 1.0 - clamp(target_point_distance, 0.0, 1.0);

    return clamp(1.0 - close - 0.5 * mid - 0.25 * far, 0.0, 1.0);
}

fn ambient_occlusion(world_position: vec4<f32>, world_normal: vec3<f32>) -> f32 {
    let fwd = world_normal;

    var sign = -1.0;
    if (fwd.z >= 0.0) {
        sign = 1.0;
    }
    let a = -1.0 / (fwd.z + sign);
    let b = fwd.x * fwd.y * a;
    let up = vec3<f32>(1.0 + sign * fwd.x * fwd.x * a, sign * b, -sign * fwd.x);
    let right = vec3<f32>(-b, -sign - fwd.y * fwd.y * a, fwd.y);

    let side = 0.3;
    let ratio = sqrt(1.0 + 2.0*side*side);
    let fwd = fwd * ratio;
    let right = right * ratio;
    let up = up * ratio;
 
    var sdf_ao = 0.5 + 
        sdf_occlusion(world_position, world_normal, 1.0) * 0.2 + 
        sdf_occlusion(world_position, fwd + up * side + right * side, 1.0) * 0.075 + 
        sdf_occlusion(world_position, fwd + up * side - right * side, 1.0) * 0.075 + 
        sdf_occlusion(world_position, fwd - up * side + right * side, 1.0) * 0.075 + 
        sdf_occlusion(world_position, fwd - up * side - right * side, 1.0) * 0.075;

    return sdf_ao;
}

fn specular_occlusion(world_position: vec4<f32>, world_normal: vec3<f32>, view: vec3<f32>) -> f32 {
    let use_direction = reflect(-view, world_normal);
 
    let near = sdf_occlusion(world_position, use_direction, 1.0);
    let mid = sdf_occlusion(world_position, use_direction * 4.0, 0.25);
    let far = sdf_occlusion(world_position, use_direction * 8.0, 0.125);

    let small = min(near, min(mid, far));
    let large = max(near, max(mid, far));
    let med = near + mid + far - small - large;

    var sdf_ao = small * 0.5 + med * 0.3 + large * 0.2;

    return sdf_ao;
}

fn ambient_light(
    world_position: vec4<f32>, 
    world_normal: vec3<f32>, 
    V: vec3<f32>, 
    diffuse_color: vec3<f32>, 
    specular_color: vec3<f32>, 
    perceptual_roughness: f32,
    occlusion: f32,
) -> vec3<f32> {
    let NdotV = max(dot(world_normal, V), 0.0001);

    let diffuse_occ = ambient_occlusion(world_position, world_normal);
    let specular_occ = specular_occlusion(world_position, world_normal, V);
    let diffuse_ambient = EnvBRDFApprox(diffuse_color, 1.0, NdotV) * diffuse_occ;
    let specular_ambient = clamp(EnvBRDFApprox(specular_color, perceptual_roughness, NdotV), vec3<f32>(0.0), vec3<f32>(1.0)) * diffuse_occ * specular_occ;

    return (diffuse_ambient + specular_ambient) * lights.ambient_color.rgb * occlusion;
}