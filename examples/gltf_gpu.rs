use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    // log::LogSettings,
    prelude::*,
    window::PresentMode,
};
use mesh2sdf::{shader::SimpleTextureMaterial, create_sdf_image, compute::{SdfComputeRequests, SdfComputePlugin}};
use mesh2sdf::{
    controller::{CameraController, ControllerPlugin},
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
    app.add_plugin(SdfComputePlugin);
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
    commands
        .spawn_bundle(TransformBundle::default())
        .insert(Rotate)
        .with_children(|p| {
            p.spawn_bundle(SceneBundle {
                scene,
                ..Default::default()
            });
        });
}

fn system(
    mut commands: Commands,
    mesh_ents: Query<(Entity, &Handle<Mesh>), Without<Sdf>>,
    meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut render_sdfs: ResMut<Assets<SdfMaterial>>,
    mut req_sdfs: ResMut<SdfComputeRequests>,
    reques: Query<Entity, (With<Sdf>, Without<Handle<SdfMaterial>>)>,
) {
    for (ent, h) in mesh_ents.iter() {
        if let Some(mesh) = meshes.get(h) {
            println!("triangles: {}", match mesh.indices() {
                Some(i) => i.len(),
                None => mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len(),
            } / 3);
            let aabb = mesh.compute_aabb().unwrap();
            println!("aabb: {:?}", aabb);
            let max_dim = aabb.half_extents.max_element();
            let mult = std::env::args().nth(2).unwrap_or("4".into()).parse::<u32>().unwrap();
            let dimensions = (aabb.half_extents / max_dim * 32.0).as_uvec3() * mult;
            println!("dimensions: {}", dimensions);

            let image = images.add(create_sdf_image(dimensions));
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
                .spawn_bundle(Camera3dBundle {
                    transform: Transform::from_translation((aabb.center + aabb.half_extents * 2.0).into()).looking_at(aabb.center.into(), Vec3::Y),
                    ..Default::default()
                })
                .insert(CameraController::default());

            commands
                .entity(ent)
                .insert(Sdf {
                    image: image.clone(),
                    aabb: aabb.clone(),
                })
                // .insert(Rotate)
                .with_children(|p| {
                    p.spawn_bundle((Sdf { image, aabb }, render));
                });
        
            req_sdfs.0.push(ent);
        }
    }

    for ent in reques.iter() {
        // rerun every frame
        req_sdfs.0.push(ent);
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

#[derive(Component)]
struct Rotate;

fn rotate(
    mut q: Query<&mut Transform, With<Rotate>>,
    time: Res<Time>,
) {
    for mut t in q.iter_mut() {
        t.rotation = Quat::from_euler(EulerRot::YXZ, time.seconds_since_startup().sin() as f32, 0.0, 0.0);
    }
}