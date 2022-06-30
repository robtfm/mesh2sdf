use bevy::{prelude::*, gltf::GltfPlugin};
use mesh2sdf::create_sdf_from_mesh_cpu;

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
    app.add_plugin(GltfPlugin);
    app.insert_resource(AmbientLight{ color: Color::WHITE, brightness: 1.0 });

    app.add_startup_system(setup);
    app.add_system(system);
    app.run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    let scene = asset_server.load("teapot.glb#Scene0");
    commands.spawn_bundle(SceneBundle{
        scene,
        ..Default::default()
    });

    commands.spawn_bundle(Camera3dBundle{
        transform: Transform::from_xyz(10.0, 10.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    });
}

fn system(
    transforms: Query<(Entity, &Transform)>,
    mesh_ents: Query<&Handle<Mesh>>,
    meshes: ResMut<Assets<Mesh>>,
) {
    println!("mesh count: {}", mesh_ents.iter().count());
    println!("transform count: {}", transforms.iter().count());

    for h in mesh_ents.iter() {
        if let Some(mesh) = meshes.get(h) {
            println!("triangles: {}", mesh.indices().unwrap().len() / 3);
            // assert!(false);
            let aabb = mesh.compute_aabb().unwrap();
            println!("aabb: {:?}", aabb);
            // let dimensions = UVec3::new(2, 1, 1);
            let dimensions = UVec3::new(128, 74, 89);
            let start = std::time::Instant::now();
            let sdf = create_sdf_from_mesh_cpu(&mesh, &aabb, dimensions);
            let end = std::time::Instant::now();
            println!("{:?}", end - start);

            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    let x = (dimensions.x+1)/2;
                    let ix = x*dimensions.z*dimensions.y+(dimensions.y-y-1)*dimensions.z+z;
                    let byte_ix = (ix*4) as usize;
                    let float = f32::from_le_bytes(sdf.data[byte_ix..byte_ix+4].try_into().unwrap());
                    if float <= 0.0 {
                        // let point = aabb.min() + scale * ((UVec3::new(x as u32,y as u32,z as u32) * 2) + 1).as_vec3a();
                        // print!("{}:{}  ", point, float);
                        print!("x");
                    } else {
                        print!(" ");
                    }
                }
                println!("");
            }

            assert!(false);
        }
    }
}
