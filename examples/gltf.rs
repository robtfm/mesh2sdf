use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    // log::LogSettings,
    prelude::*,
    window::PresentMode,
};
use mesh2sdf::shader::SimpleTextureMaterial;
use mesh2sdf::{
    controller::{CameraController, ControllerPlugin},
    create_sdf_from_mesh_cpu,
    render::{SdfMaterial, SdfMaterialSpec, SdfRenderPlugin},
    Sdf,
};

fn main() {
    let mut app = App::new();
    // app.insert_resource(LogSettings {
    //     level: bevy::log::Level::DEBUG,
    //     filter: "wgpu=error".to_string(),
    // });
    app.insert_resource(WindowDescriptor {
        present_mode: PresentMode::Immediate,
        ..Default::default()
    });
    app.add_plugin(LogDiagnosticsPlugin::default());
    app.add_plugin(FrameTimeDiagnosticsPlugin::default());
    app.add_plugins(DefaultPlugins);
    app.add_plugin(SdfRenderPlugin);
    app.add_plugin(ControllerPlugin);
    app.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 1.0,
    });

    app.add_startup_system(setup);
    app.add_system(system);
    app.add_system(toggle);
    app.add_system(rotate);
    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let filename = std::env::args().nth(1).unwrap_or("teapot".into());

    let scene = asset_server.load(format!("gltf/{}.glb#Scene0", filename).as_str());
    commands.spawn_bundle(SceneBundle {
        scene,
        transform: Transform::from_xyz(3.0, 0.5, -5.0),
        ..Default::default()
    });

    commands
        .spawn_bundle(Camera3dBundle {
            transform: Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..Default::default()
        })
        .insert(CameraController::default());
}

fn system(
    mut commands: Commands,
    mesh_ents: Query<(Entity, &Handle<Mesh>, &GlobalTransform), Without<Sdf>>,
    meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut render_sdfs: ResMut<Assets<SdfMaterial>>,
) {
    for (ent, h, transform) in mesh_ents.iter() {
        if transform.translation == Vec3::ZERO {
            continue;
        }
        if let Some(mesh) = meshes.get(h) {
            println!("triangles: {}", mesh.indices().unwrap().len() / 3);
            let mut aabb = mesh.compute_aabb().unwrap();
            aabb.half_extents *= 1.1;
            println!("aabb: {:?}", aabb);
            let max_dim = aabb.half_extents.max_element();
            let mult = std::env::args().nth(2).unwrap_or("4".into()).parse::<u32>().unwrap();
            let dimensions = (aabb.half_extents / max_dim * 32.0).as_uvec3() * mult;
            println!("dimensions: {}", dimensions);
            // let dimensions = UVec3::new(128, 74, 89);
            let start = std::time::Instant::now();
            let sdf = create_sdf_from_mesh_cpu(&mesh, &aabb, dimensions, None);//, Some(UVec3::new(0, 63-0, 63)));
            let end = std::time::Instant::now();
            println!("{:?}", end - start);

            // let mut p = true;

            // for x in 0..dimensions.x {
            //     for y in 0..dimensions.y {
            //         for z in 0..dimensions.z {
            //             let ix =
            //                 z * dimensions.x * dimensions.y + (dimensions.y - y - 1) * dimensions.x + x;
            //             let byte_ix = (ix * 4) as usize;
            //             let float =
            //                 f32::from_le_bytes(sdf.data[byte_ix..byte_ix + 4].try_into().unwrap());
            //             if float <= 0.0 {
            //                 if z > 54 || p { 
            //                     // println!("BAD: {:?}: {}", [x, y, z], float);
            //                     p = false;
            //                     // assert!(false);
            //                 }
            //                 // let point = aabb.min() + scale * ((UVec3::new(x as u32,y as u32,z as u32) * 2) + 1).as_vec3a();
            //                 // print!("{}:{}  ", point, float);
            //                 print!("x");
            //             } else {
            //                 print!(" ");
            //             }
            //         }
            //         println!("");
            //     }
            // }

            let image = images.add(sdf);
            let render = render_sdfs.add(SimpleTextureMaterial(SdfMaterialSpec {
                aabb: aabb.clone(),
                image: image.clone(),
                min_step_size: 0.001,
                hit_threshold: -0.0001,
                max_step_count: 200,
                // hit_color: Color::NONE,
                // step_color: Color::NONE,
                ..Default::default()
            }));

            commands
                .entity(ent)
                .insert(Sdf {
                    image: image.clone(),
                    aabb: aabb.clone(),
                })
                .with_children(|p| {
                    p.spawn_bundle((Sdf { image, aabb }, render));
                });
        }
    }
}

fn toggle(
    key_input: Res<Input<KeyCode>>,
    mut sdf: Query<&mut Visibility, (With<Handle<SdfMaterial>>, Without<Handle<StandardMaterial>>)>,
    mut base: Query<&mut Visibility, With<Handle<StandardMaterial>>>,
) {
    if key_input.just_pressed(KeyCode::O) {
        for mut vis in sdf.iter_mut() {
            vis.is_visible = !vis.is_visible;
        }
    }
    if key_input.just_pressed(KeyCode::P) {
        for mut vis in base.iter_mut() {
            vis.is_visible = !vis.is_visible;
        }
    }
}

fn rotate(
    mut q: Query<&mut Transform, (With<Handle<Mesh>>, With<Handle<StandardMaterial>>)>,
    time: Res<Time>,
) {
    for mut t in q.iter_mut() {
        t.rotation = Quat::from_euler(EulerRot::YXZ, time.seconds_since_startup().sin() as f32, 0.0, 0.0);
    }
}