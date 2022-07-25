use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    prelude::*,
};
use mesh2sdf::{
    controller::{CameraController, ControllerPlugin},
    debug_render::{SdfRender, SdfRenderPlugin},
    Sdf, SdfAtlas, SdfGlobalSettings, SdfPlugin,
};

#[allow(unused_imports)]
use bevy::{log::LogSettings, window::PresentMode};

fn main() {
    let mut app = App::new();

    // app.insert_resource(LogSettings {
    //     level: bevy::log::Level::DEBUG,
    //     filter: "wgpu=error".to_string(),
    // });

    // app.insert_resource(WindowDescriptor {
    //     present_mode: PresentMode::Immediate,
    //     ..Default::default()
    // });

    app.insert_resource(SdfGlobalSettings {
        atlas_page_size: UVec3::splat(400),
        buffer_size: 15.0,
        unit_size: 5.0,
        ambient_distance: 15.0,
    });

    SdfPlugin::add_view_bindings(&mut app);
    app.add_plugin(LogDiagnosticsPlugin::default());
    app.add_plugin(FrameTimeDiagnosticsPlugin::default());
    app.add_plugins(DefaultPlugins)
        .add_plugin(SdfPlugin)
        .add_plugin(SdfRenderPlugin)
        .add_plugin(ControllerPlugin)
        .insert_resource(ClearColor(Color::rgb(0.7, 0.7, 1.0)))
        .add_startup_system(setup)
        .add_system(movement)
        .add_system(finalise_scene)
        .add_system(toggle)
        .run();
}

#[derive(Component)]
struct Movable;

struct Animations(Vec<Handle<AnimationClip>>);

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // ground plane
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Plane { size: 500.0 })),
            material: materials.add(StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 1.0,
                ..default()
            }),
            ..default()
        })
        .insert(Sdf::new_scaled(1.0)); // min size

    // left wall
    let mut transform = Transform::from_xyz(125.0, 125.0, 0.0);
    transform.rotate_z(std::f32::consts::FRAC_PI_2);
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Box::new(250.0, 50.0, 250.0))),
            transform,
            material: materials.add(StandardMaterial {
                base_color: Color::ANTIQUE_WHITE,
                perceptual_roughness: 1.0,
                ..default()
            }),
            ..default()
        })
        .insert(Sdf::new_scaled(1.0)); // min size

    // back (right) wall
    let mut transform = Transform::from_xyz(0.0, 125.0, -125.0);
    transform.rotate_x(std::f32::consts::FRAC_PI_2);
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Box::new(250.0, 50.0, 250.0))),
            transform,
            material: materials.add(StandardMaterial {
                base_color: Color::ANTIQUE_WHITE,
                perceptual_roughness: 1.0,
                ..default()
            }),
            ..default()
        })
        .insert(Sdf::new_scaled(1.0)); // min size

    // cube
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 50.0 })),
            material: materials.add(StandardMaterial {
                base_color: Color::PINK,
                ..default()
            }),
            transform: Transform::from_xyz(-1.9, 0.5, -1.9),
            ..default()
        })
        .insert(Sdf::new_scaled(1.0));

    // sphere
    commands
        .spawn_bundle(PbrBundle {
            mesh: meshes.add(Mesh::from(shape::UVSphere {
                radius: 25.0,
                ..default()
            })),
            material: materials.add(StandardMaterial {
                base_color: Color::LIME_GREEN,
                ..default()
            }),
            transform: Transform::from_xyz(75.0, 50.0, 75.0),
            ..default()
        })
        // .insert(Movable)
        .insert(Sdf::new_scaled(1.0));

    // fox
    let scene = asset_server.load("gltf/fox.glb#Scene0");
    commands
        .spawn_bundle(SpatialBundle {
            // transform: Transform::from_scale(Vec3::splat(0.02)),
            ..Default::default()
        })
        // .insert(Rotate)
        .with_children(|p| {
            p.spawn_bundle(SceneBundle {
                scene,
                transform: Transform::from_xyz(50.0, 0.0, 75.0),
                ..Default::default()
            });
        })
        .insert(Movable);
    commands.insert_resource(Animations(vec![
        asset_server.load("gltf/fox.glb#Animation0")
    ]));

    // ambient light
    commands.insert_resource(AmbientLight {
        color: Color::rgb(0.7, 0.7, 1.0),
        brightness: 0.4,
    });

    // camera
    commands
        .spawn_bundle(Camera3dBundle {
            transform: Transform::from_xyz(-100.0, 125.0, 250.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        })
        .insert(CameraController{ walk_speed: 250.0, ..Default::default()});
}

fn finalise_scene(
    mut commands: Commands,
    q: Query<(Entity, &Handle<StandardMaterial>), (With<Handle<Mesh>>, Without<Sdf>)>,

    animations: Res<Animations>,
    mut player: Query<&mut AnimationPlayer>,
    mut done: Local<bool>,
    mut mats: ResMut<Assets<StandardMaterial>>,
) {
    for (ent, m) in q.iter() {
        commands
            .entity(ent)
            .insert(Sdf::new_scaled(1.0))
            .insert(Movable);
        if let Some(mat) = mats.get_mut(m) {
            mat.base_color = Color::ORANGE_RED;
            mat.metallic = 0.01;
            mat.reflectance = 0.5;
            mat.perceptual_roughness = 0.089;
        }
    }

    if !*done {
        if let Ok(mut player) = player.get_single_mut() {
            player.play(animations.0[0].clone_weak()).repeat();
            *done = true;
        }
    }
}

fn toggle(
    input: Res<Input<KeyCode>>,
    sdfs: Query<Entity, With<Sdf>>,
    mut commands: Commands,
    mut atlas: ResMut<SdfAtlas>,
) {
    if input.just_pressed(KeyCode::O) {
        for ent in sdfs.iter() {
            commands.entity(ent).with_children(|p| {
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

        atlas.page.purge_all();
    }
}

fn movement(
    input: Res<Input<KeyCode>>,
    time: Res<Time>,
    mut query: Query<&mut Transform, With<Movable>>,
) {
    for mut transform in &mut query {
        let mut direction = Vec3::ZERO;
        let mut scale = 1.0;
        if input.pressed(KeyCode::Up) {
            direction.y += 1.0;
        }
        if input.pressed(KeyCode::Down) {
            direction.y -= 1.0;
        }
        if input.pressed(KeyCode::Left) {
            direction.x -= 1.0;
        }
        if input.pressed(KeyCode::Right) {
            direction.x += 1.0;
        }

        if input.pressed(KeyCode::Key9) {
            scale *= 0.99;
        }
        if input.pressed(KeyCode::Key0) {
            scale *= 1.01;
        }

        transform.translation += time.delta_seconds() * 100.0 * direction;
        transform.scale *= scale;
    }
}
