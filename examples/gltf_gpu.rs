#![feature(let_else)]
use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    // log::LogSettings,
    prelude::*,
    render::primitives::Aabb,
    window::PresentMode,
};
use mesh2sdf::{
    controller::{CameraController, ControllerPlugin},
    debug_render::{SdfMaterial, SdfRenderPlugin},
    Sdf, SdfAtlas,
};
use mesh2sdf::{debug_render::SdfRender, SdfGenMode, SdfGlobalSettings, SdfPlugin};

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
    SdfPlugin::add_view_bindings(&mut app);
    app.add_plugin(LogDiagnosticsPlugin::default());
    app.add_plugin(FrameTimeDiagnosticsPlugin::default());
    app.add_plugins(DefaultPlugins);

    let unit_size = std::env::args()
        .nth(2)
        .unwrap_or("1".into())
        .parse::<f32>()
        .unwrap();

    let buffer_size = std::env::args()
        .nth(3)
        .unwrap_or("1".into())
        .parse::<f32>()
        .unwrap();

    app.insert_resource(SdfGlobalSettings {
        buffer_size,
        unit_size,
        atlas_page_size: UVec3::splat(400),
        ambient_distance: 1.0,
    });
    app.add_plugin(SdfPlugin);
    app.add_plugin(SdfRenderPlugin);
    app.add_plugin(ControllerPlugin);
    app.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 1.0,
    });

    app.add_startup_system(setup);
    app.add_system(setup_scene_once_loaded);
    app.add_system(system);
    app.add_system(toggle);
    app.add_system(rotate);
    app.run();
}

struct Animations(Vec<Handle<AnimationClip>>);

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let filename = std::env::args().nth(1).unwrap_or("teapot".into());

    // always load a teapot in the sky
    let scene = asset_server.load("gltf/teapot.glb#Scene0");
    commands
        .spawn_bundle(SpatialBundle::default())
        // .insert(Rotate)
        .with_children(|p| {
            p.spawn_bundle(SceneBundle {
                scene,
                transform: Transform::from_xyz(1.0, 5.5, 3.0),
                ..Default::default()
            });
        });

    let scene = asset_server.load(format!("gltf/{}.glb#Scene0", filename).as_str());
    commands
        .spawn_bundle(SpatialBundle::default())
        // .insert(Rotate)
        .with_children(|p| {
            p.spawn_bundle(SceneBundle {
                scene,
                transform: Transform::from_xyz(1.0, 1.5, 3.0),
                ..Default::default()
            });
        });

    commands.insert_resource(Animations(vec![
        asset_server.load(format!("gltf/{}.glb#Animation0", filename).as_str())
    ]));
}

fn setup_scene_once_loaded(
    animations: Res<Animations>,
    mut player: Query<&mut AnimationPlayer>,
    mut done: Local<bool>,
) {
    if !*done {
        if let Ok(mut player) = player.get_single_mut() {
            player.play(animations.0[0].clone_weak()).repeat();
            *done = true;
        }
    }
}

fn system(
    mut commands: Commands,
    mesh_ents: Query<(Entity, &Handle<Mesh>, &Aabb), (Without<SdfRender>, Without<Sdf>)>,
    meshes: Res<Assets<Mesh>>,
    mut cam_added: Local<bool>,
) {
    for (ent, h, aabb) in mesh_ents.iter() {
        if let Some(mesh) = meshes.get(h) {
            println!(
                "triangles: {}",
                match mesh.indices() {
                    Some(i) => i.len(),
                    None => mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap().len(),
                } / 3
            );

            if !*cam_added {
                commands
                    .spawn_bundle(Camera3dBundle {
                        transform: Transform::from_translation(
                            (aabb.center + aabb.half_extents * 5.0).into(),
                        )
                        .looking_at(aabb.center.into(), Vec3::Y),
                        ..Default::default()
                    })
                    .insert(CameraController::default());
                *cam_added = true;
            }

            commands
                .entity(ent)
                .insert(Sdf {
                    mode: SdfGenMode::FromPrimaryMesh,
                    ..Default::default()
                })
                // .insert(Rotate)
                .with_children(|p| {
                    p.spawn_bundle(SpatialBundle::default()).insert(SdfRender {
                        entity: ent,
                        base_color: Color::rgba_linear(0.0, 0.0, 0.0, 1.0),
                        hit_color: Color::rgba_linear(1.0, 0.0, 0.0, 0.0),
                        step_color: Color::rgba_linear(0.0, 1.0, 0.0, 0.0),
                        distance_color: Color::rgba_linear(0.0, 0.0, 1.0, 0.0),
                        min_step_size: 0.1,
                        hit_threshold: 0.1,
                        max_step_count: 50,
                    });
                });
        }
    }
}

fn toggle(
    key_input: Res<Input<KeyCode>>,
    mut sdf: Query<&mut Visibility, (With<Handle<SdfMaterial>>, Without<Handle<StandardMaterial>>)>,
    mut base: Query<&mut Visibility, With<Handle<StandardMaterial>>>,
    mut player: Query<&mut AnimationPlayer>,
    mut atlas: ResMut<SdfAtlas>,
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

    if key_input.just_pressed(KeyCode::Space) {
        for mut player in player.iter_mut() {
            if player.is_paused() {
                player.resume();
            } else {
                player.pause();
            }
        }
    }

    if key_input.just_pressed(KeyCode::L) {
        atlas.page.purge_all();
    }

    if key_input.just_pressed(KeyCode::K) {
        println!("{} sdfmats", sdf.iter().count());
    }
}

#[derive(Component)]
struct Rotate;

fn rotate(mut q: Query<&mut Transform, With<Rotate>>, time: Res<Time>) {
    for mut t in q.iter_mut() {
        t.rotation = Quat::from_euler(
            EulerRot::YXZ,
            time.seconds_since_startup().sin() as f32,
            0.0,
            0.0,
        );
    }
}
