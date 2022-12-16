use dot_vox::{Rotation, SceneNode};
use macroquad::experimental::collections::storage;
use macroquad::input::KeyCode::*;
use macroquad::models::Vertex;
use macroquad::prelude::*;
use macroquad::window::miniquad::*;
use nanoserde::{DeRon, SerRon};
use std::{
    collections::{HashSet, VecDeque},
    f32::consts::PI,
    fs::{self, File},
    io::prelude::*,
};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumCount, EnumIter, EnumString, EnumVariantNames, IntoStaticStr};

const VERTEX: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;

varying lowp vec2 uv;
varying lowp vec4 color;

uniform mat4 Model;
uniform mat4 Projection;
void main() {
    gl_Position = Projection * Model * vec4(position, 1);
    uv = texcoord;
    color = color0 / 255.0;
}"#;

const FRAGMENT: &str = r#"#version 100
varying lowp vec2 uv;
varying lowp vec4 color;
uniform sampler2D Texture;
uniform lowp vec3 face_normal;
void main() {
    lowp vec3 lightPos = vec3(100.0, -100.0, 100.0);
    lowp vec3 lightColor = vec3(0.8, 0.8, 1.0);
    lowp float ambientStrength = 0.5;
    lowp vec3 ambient = ambientStrength * lightColor;
    lowp vec3 norm = normalize(face_normal);
    lowp vec3 lightDir = normalize(lightPos - gl_FragCoord.xyz);  
    lowp float diff = max(dot(norm, lightDir), 0.0);
    lowp vec3 diffuse = diff * lightColor;
    lowp vec4 lighting = vec4((ambient + diffuse), 1.0);
    gl_FragColor = lighting * color * texture2D(Texture, uv);
}"#;

macro_rules! i {
    ($var: ident) => {
        info!("{} = {:?}", stringify!($var), $var);
    };
}

macro_rules! d {
    ($var: ident) => {
        debug!("{} = {:?}", stringify!($var), $var);
    };
}

macro_rules! t {
    ($var: ident) => {
        trace!("{} = {:?}", stringify!($var), $var);
    };
}

// CONSTANTS
// ------------------------------------------------------------------------------------------------
const DT: f32 = 0.1;
const GRID_SCALE: f32 = 1.;
const GRID_OFFSET: f32 = GRID_SCALE / 2.;
const GRID_SLICES: usize = 200;
const GRID_BOUNDS: f32 = GRID_SLICES as f32;
const UP: Vec3 = vec3(0., -1., 0.);
const H: f32 = 20.;
// ------------------------------------------------------------------------------------------------

// UTILITY FUNCTIONS
// ------------------------------------------------------------------------------------------------
/// projects 2d game space vector into 3d world space
#[inline]
const fn p3d(p: Vec2) -> Vec3 {
    vec3(p.x, 0.0, p.y)
}

/// projects 2d grid index into 3d world space coord at center of grid cell
#[inline]
fn g3d(g: (usize, usize)) -> Vec3 {
    vec3(g.0 as f32 + GRID_OFFSET, 0., g.1 as f32 + GRID_OFFSET)
}

fn draw_info_line(anchor: &mut Vec2, text: &str) {
    let font_size = 20.;
    anchor.y += font_size;
    draw_text(text, anchor.x, anchor.y, font_size, GREEN);
}

fn draw_target(camera: &Camera3D, p: Vec2) {
    let params = DrawTextureParams {
        dest_size: None,
        source: None,
        rotation: 0.,
        pivot: None,
        flip_x: false,
        flip_y: true,
    };
    draw_texture_ex(
        camera.render_target.unwrap().texture,
        p.x,
        p.y,
        WHITE,
        params,
    );
}

fn map_voxels(voxels: &[dot_vox::Voxel]) -> HashSet<(u8, u8, u8)> {
    let mut lut = HashSet::new();
    for v in voxels {
        lut.insert((v.x, v.y, v.z));
    }
    lut
}

// NOTE: For no good reason in particular, currently 'Front' is +Z, 'Top' is +Y, and 'Right' is +X
// in 'voxel space'
const FRONT: usize = 0;
const BACK: usize = 1;
const TOP: usize = 2;
const BOTTOM: usize = 3;
const LEFT: usize = 4;
const RIGHT: usize = 5;

#[derive(Debug, Clone)]
pub struct FaceMesh {
    pub vertices: Vec<macroquad::models::Vertex>,
    pub indices: [Vec<u16>; 6],
}

fn gen_mesh(voxels: &[dot_vox::Voxel], palette: &[dot_vox::Color]) -> Vec<FaceMesh> {
    let neighbor_lut = map_voxels(voxels);
    let mut meshes = Vec::new();

    for voxel_slice in voxels.chunks(512) {
        let mut mesh = FaceMesh {
            vertices: Vec::with_capacity(voxels.len()),
            indices: [0; 6].map(|_| Vec::with_capacity(voxels.len())),
        };

        for v in voxel_slice {
            let c = palette[v.i as usize];
            let color = Color::from_rgba(c.r, c.g, c.b, c.a);
            let x = v.x as f32;
            let y = v.y as f32;
            let z = v.z as f32;

            let visible_front = match v.z {
                u8::MAX => true,
                _ => neighbor_lut.get(&(v.x, v.y, v.z + 1)).is_none(),
            };
            let visible_back = match v.z {
                u8::MIN => true,
                _ => neighbor_lut.get(&(v.x, v.y, v.z - 1)).is_none(),
            };
            let visible_top = match v.y {
                u8::MAX => true,
                _ => neighbor_lut.get(&(v.x, v.y + 1, v.z)).is_none(),
            };
            let visible_bottom = match v.y {
                u8::MIN => true,
                _ => neighbor_lut.get(&(v.x, v.y - 1, v.z)).is_none(),
            };
            let visible_left = match v.x {
                u8::MIN => true,
                _ => neighbor_lut.get(&(v.x - 1, v.y, v.z)).is_none(),
            };
            let visible_right = match v.x {
                u8::MAX => true,
                _ => neighbor_lut.get(&(v.x + 1, v.y, v.z)).is_none(),
            };

            // front vertices (bl, br, tr, tl)
            let fbl = mesh.vertices.len() as u16;
            if visible_front || visible_bottom || visible_left {
                mesh.vertices
                    .push(Vertex::new(x - 0.5, y - 0.5, z + 0.5, 0., 0., color));
            }

            let fbr = mesh.vertices.len() as u16;
            if visible_front || visible_bottom || visible_right {
                mesh.vertices
                    .push(Vertex::new(x + 0.5, y - 0.5, z + 0.5, 1., 0., color));
            }

            let ftr = mesh.vertices.len() as u16;
            if visible_front || visible_top || visible_right {
                mesh.vertices
                    .push(Vertex::new(x + 0.5, y + 0.5, z + 0.5, 1., 1., color));
            }

            let ftl = mesh.vertices.len() as u16;
            if visible_front || visible_top || visible_left {
                mesh.vertices
                    .push(Vertex::new(x - 0.5, y + 0.5, z + 0.5, 0., 1., color));
            }

            let bbl = mesh.vertices.len() as u16;
            if visible_back || visible_bottom || visible_left {
                mesh.vertices
                    .push(Vertex::new(x - 0.5, y - 0.5, z - 0.5, 0., 0., color));
            }

            let bbr = mesh.vertices.len() as u16;
            if visible_back || visible_bottom || visible_right {
                mesh.vertices
                    .push(Vertex::new(x + 0.5, y - 0.5, z - 0.5, 1., 0., color));
            }

            let btr = mesh.vertices.len() as u16;
            if visible_back || visible_top || visible_right {
                mesh.vertices
                    .push(Vertex::new(x + 0.5, y + 0.5, z - 0.5, 1., 1., color));
            }

            let btl = mesh.vertices.len() as u16;
            if visible_back || visible_top || visible_left {
                mesh.vertices
                    .push(Vertex::new(x - 0.5, y + 0.5, z - 0.5, 0., 1., color));
            }
            if visible_front {
                mesh.indices[FRONT].extend([fbl, fbr, ftr, fbl, ftr, ftl]);
            }
            if visible_back {
                mesh.indices[BACK].extend([bbl, bbr, btr, bbl, btr, btl]);
            }
            if visible_top {
                mesh.indices[TOP].extend([ftl, ftr, btr, ftl, btr, btl]);
            }
            if visible_bottom {
                mesh.indices[BOTTOM].extend([bbl, bbr, fbr, bbl, fbr, fbl]);
            }
            if visible_left {
                mesh.indices[LEFT].extend([bbl, fbl, ftl, bbl, ftl, btl]);
            }
            if visible_right {
                mesh.indices[RIGHT].extend([bbr, fbr, ftr, bbr, ftr, btr]);
            }
        }
        meshes.push(mesh);
    }
    meshes
}

pub fn draw_face_mesh_list(mesh_list: &[FaceMesh]) {
    let gl = unsafe { get_internal_gl().quad_gl };
    let mat = storage::get::<FlatMat>().0;
    gl_use_material(mat);
    for mesh in mesh_list {
        for (idx, face_indices) in mesh.indices.iter().enumerate() {
            let norm = match idx {
                FRONT => Vec3::Z,
                BACK => Vec3::NEG_Z,
                TOP => Vec3::Y,
                BOTTOM => Vec3::NEG_Y,
                LEFT => Vec3::NEG_X,
                RIGHT => Vec3::X,
                _ => std::unreachable!(),
            };
            mat.set_uniform("face_normal", norm);
            gl.texture(None);
            gl.draw_mode(DrawMode::Triangles);
            gl.geometry(&mesh.vertices[..], &face_indices[..]);
        }
    }
    gl_use_default_material();
}

pub fn draw_scene_recursive(
    meshes: &[Vec<FaceMesh>],
    scene: &dot_vox::DotVoxData,
    node_idx: u32,
    parent: Option<u32>,
    translation: IVec3,
    rotation: dot_vox::Rotation,
) {
    let node = &scene.scenes[node_idx as usize];
    let gl = unsafe { get_internal_gl().quad_gl };

    match node {
        SceneNode::Transform {
            attributes: _,
            frames,
            child,
            layer_id: _,
        } => {
            let mut this_translation = frames[0]
                .position()
                .map(|position| IVec3 {
                    x: position.x,
                    y: position.y,
                    z: position.z,
                })
                .unwrap_or(IVec3::ZERO);

            let this_rotation = frames[0].orientation().unwrap_or(Rotation::IDENTITY);
            let translation = translation + this_translation;
            let rotation = rotation * this_rotation;

            draw_scene_recursive(meshes, scene, *child, Some(node_idx), translation, rotation);
        }

        SceneNode::Group {
            attributes: _,
            children,
        } => {
            let mut translation = translation.as_vec3().xzy();
            translation.z *= -1.0;

            let (quat, scale) = rotation.to_quat_scale();
            let quat = glam::Quat::from_array(quat);
            let scale = -glam::Vec3::from_array(scale).xzy();

            let size = Vec3::ZERO; // only apply centering on final translation!
            let center = quat * size / 2.0;
            let mat =
                Mat4::from_scale_rotation_translation(scale, quat, translation - center * scale);
            gl.push_model_matrix(mat);
            for child in children {
                draw_scene_recursive(
                    meshes,
                    scene,
                    *child,
                    Some(node_idx),
                    IVec3::ZERO,
                    Rotation::IDENTITY,
                );
            }
            gl.pop_model_matrix();
        }

        SceneNode::Shape {
            attributes: _,
            models,
        } => {
            let mut translation = translation.as_vec3().xzy();
            translation.z *= -1.0;

            let (quat, scale) = rotation.to_quat_scale();
            let quat = glam::Quat::from_array(quat);
            let scale = -glam::Vec3::from_array(scale).xzy();

            for model in models {
                let idx = model.model_id;
                let sz = scene.models[idx as usize].size;
                let size = vec3(sz.x as f32, sz.z as f32, sz.y as f32);
                let center = quat * size / 2.0;
                let mat = Mat4::from_scale_rotation_translation(
                    scale,
                    quat,
                    translation - center * scale,
                );
                gl.push_model_matrix(mat);
                draw_face_mesh_list(&meshes[idx as usize]);
                gl.pop_model_matrix();
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------

const KEY_BINDS: [(KeyCode, &'static str); 10] = [
    (Key0, "0"),
    (Key1, "1"),
    (Key2, "2"),
    (Key3, "3"),
    (Key4, "4"),
    (Key5, "5"),
    (Key6, "6"),
    (Key7, "7"),
    (Key8, "8"),
    (Key9, "9"),
];

#[derive(SerRon, DeRon, Default)]
struct Ship {
    p: Vec2,
    v: Vec2,
    f: Vec2,
}

#[derive(
    Default,
    Debug,
    SerRon,
    DeRon,
    Display,
    EnumString,
    EnumCount,
    EnumIter,
    EnumVariantNames,
    IntoStaticStr,
    Clone,
)]
enum Dir {
    #[default]
    E,
    N,
    W,
    S,
}

impl Dir {
    const fn v(&self) -> Vec2 {
        match self {
            Dir::N => Vec2::Y,
            Dir::S => Vec2::NEG_Y,
            Dir::E => Vec2::X,
            Dir::W => Vec2::NEG_X,
        }
    }

    fn from_vec2(v: Vec2) -> Self {
        let compass = (f32::round(f32::atan2(v.y, v.x) / (2. * PI / 4.)) + 4.) as usize % 4;
        match compass {
            0 => Dir::E,
            1 => Dir::N,
            2 => Dir::W,
            3 => Dir::S,
            _ => unreachable!(),
        }
    }
}

#[derive(
    Default,
    Debug,
    SerRon,
    DeRon,
    Display,
    EnumString,
    EnumCount,
    EnumIter,
    EnumVariantNames,
    IntoStaticStr,
    PartialEq,
    Eq,
    Clone,
)]
enum Circuit {
    #[default]
    None,
    Wire,
    Head,
    Tail,
}

#[derive(
    Default,
    Debug,
    SerRon,
    DeRon,
    Display,
    EnumString,
    EnumCount,
    EnumIter,
    EnumVariantNames,
    IntoStaticStr,
    Clone,
)]
enum ShipTile {
    #[default]
    Ground,
    Wall,
    CenterOfMass,
    Engine(Dir, bool),
    Wire,
    Input(usize),
}

#[derive(
    Default,
    Clone,
    SerRon,
    DeRon,
    Display,
    EnumString,
    EnumCount,
    EnumIter,
    EnumVariantNames,
    IntoStaticStr,
)]
enum Terrain {
    #[default]
    None,
    Asteroid {
        p: Vec2,
        v: Vec2,
        f: Vec2,
    },
}

fn to_screen(p: Vec3, camera: &Camera3D) -> Vec2 {
    let coord = camera.matrix().project_point3(p).xy() * vec2(1.0, -1.0);
    let window = vec2(screen_width(), screen_height());
    let res = coord * window / 2. + window / 2.;
    res
}

/// Initial circuit state from a grid
fn init_circuit(grid: &Vec<Vec<ShipTile>>) -> Vec<Vec<Circuit>> {
    let mut state = vec![vec![Circuit::None; grid.len()]; grid.len()];
    for (x, col) in grid.iter().enumerate() {
        for (y, tile) in col.iter().enumerate() {
            state[x][y] = match tile {
                ShipTile::Wire | ShipTile::Input(_) => Circuit::Wire,
                _ => Circuit::None,
            };
        }
    }
    state
}

fn window_conf() -> Conf {
    Conf {
        window_title: "jspace".to_owned(),
        fullscreen: false,
        window_width: 1920,
        window_height: 1080,
        window_resizable: false,
        ..Default::default()
    }
}

struct FlatMat(Material);

#[macroquad::main(window_conf)]
async fn main() {
    env_logger::init();
    //set_cursor_grab(true);

    // CONFIGURATIONS
    let scroll_sens = 0.1;
    let pan_sens: f32 = 0.1;
    let look_sens = 0.1;

    // MODELS
    let ship_vox = dot_vox::load("data/models/cargo-spaceship-by-fps-agency.vox").unwrap();
    let ship_meshes: Vec<Vec<FaceMesh>> = ship_vox
        .models
        .iter()
        .map(|m| {
            let num = m.voxels.len();
            i!(num);
            gen_mesh(&m.voxels, &ship_vox.palette)
        })
        .collect();

    storage::store(FlatMat(
        load_material(
            VERTEX,
            FRAGMENT,
            MaterialParams {
                uniforms: vec![("face_normal".to_string(), UniformType::Float3)],
                pipeline_params: PipelineParams {
                    depth_test: Comparison::LessOrEqual,
                    depth_write: true,
                    color_blend: Some(BlendState::new(
                        Equation::Add,
                        BlendFactor::Value(BlendValue::SourceAlpha),
                        BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                    )),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap(),
    ));

    // EDITOR ELEMENTS
    let mut ship_files: Vec<_> = fs::read_dir("data/ships")
        .unwrap()
        .map(|p| p.unwrap().path())
        .filter(|p| match p.extension() {
            Some(ext) => ext == "ship",
            None => false,
        })
        .collect();
    ship_files.sort_by(|a, b| {
        a.as_path()
            .to_str()
            .unwrap()
            .to_lowercase()
            .cmp(&b.as_path().to_str().unwrap().to_lowercase())
    });

    let hotload_ship = |idx: usize| {
        let grid: Vec<Vec<ShipTile>> = DeRon::deserialize_ron(
            fs::read_to_string(ship_files[idx].as_path())
                .unwrap()
                .as_str(),
        )
        .unwrap();
        grid
    };
    let mut tile_type_iter = ShipTile::iter().skip(1).cycle(); // loop through enum types, but skip
    let mut grid = hotload_ship(0);
    let mut circuit_state_a = init_circuit(&grid);
    let mut circuit_state_b = circuit_state_a.clone();

    // WORLD ELEMENTS
    let mut ship = Ship::default();
    let map = Vec::<Terrain>::new();

    // INPUT ELEMENTS
    let mut pos = vec3(0., 0., 0.);
    let mut yaw: f32 = 1.18;
    let mut pitch: f32 = 0.0;
    let mut front = vec3(
        yaw.cos() * pitch.cos(),
        pitch.sin(),
        yaw.sin() * pitch.cos(),
    )
    .normalize();
    let mut right;
    let mut fps_up = UP;
    let mut last_mouse_pos: Vec2 = mouse_position().into();

    let mut editor_pos = vec2(grid.len() as f32 / 2., grid.len() as f32 / 2.);

    let mut zoom = 1.0;
    let mut active_ship_file: usize = 0;
    let mut new_tile_type: ShipTile = tile_type_iter.next().unwrap();
    let mut selected_tile: Option<(usize, usize)> = None;

    let mut keybind_idx = 0;
    let mut state_count = 0;

    loop {
        let frame_delta = get_frame_time();

        // PREPARE FRAME
        clear_background(DARKGRAY);
        let editor_camera = Camera3D {
            position: vec3(editor_pos.x, -15. * zoom, editor_pos.y),
            up: UP,
            target: vec3(editor_pos.x, 0., editor_pos.y + 1.0),
            render_target: Some(render_target(
                screen_width() as u32 / 2,
                screen_height() as u32 / 2,
            )),
            ..Default::default()
        };
        let world_camera = Camera3D {
            position: pos,
            up: fps_up,
            target: pos + front,
            // render_target: Some(render_target(screen_width() as u32, screen_height() as u32)),
            ..Default::default()
        };

        set_camera(&editor_camera);

        // USER INPUT
        let mouse_delta: Vec2 = Vec2::from(mouse_position()) - last_mouse_pos;
        last_mouse_pos = mouse_position().into();
        if is_key_down(Space) {
            yaw += -mouse_delta.x * frame_delta * look_sens;
            pitch += mouse_delta.y * frame_delta * look_sens;
            pitch = f32::clamp(pitch, -1.5, 1.5);
        }
        front = vec3(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
        .normalize();
        right = front.cross(UP).normalize();
        fps_up = right.cross(front).normalize();

        let mouse_world: Vec2 = {
            // SOURCE: https://antongerdelan.net/opengl/raycasting.html
            let clip = (mouse_position_local() * vec2(1.0, -1.0))
                .extend(-1.0)
                .extend(1.0);
            let eye = (editor_camera.proj().inverse() * clip)
                .xy()
                .extend(-1.0)
                .extend(0.0);
            let ray = (editor_camera.view().inverse() * eye).xyz().normalize();

            let distance_to_ground = -editor_camera.position.dot(UP) / (ray.dot(UP));
            (editor_camera.position + ray * distance_to_ground).xz()
        };
        draw_sphere(p3d(mouse_world), 0.1, None, GREEN);

        let mouse_grid: Option<(usize, usize)> = {
            let is_inside_grid_bounds = mouse_world.x >= 0.
                && mouse_world.y >= 0.
                && mouse_world.x <= GRID_BOUNDS
                && mouse_world.y <= GRID_BOUNDS;

            if is_inside_grid_bounds {
                Some((mouse_world.x as usize, mouse_world.y as usize))
            } else {
                None
            }
        };
        if let Some(mouse_grid) = mouse_grid {
            draw_sphere(g3d(mouse_grid), 0.1, None, BLUE);
        }

        // INPUT_CAMERA
        match mouse_wheel() {
            (_x, y) if y != 0.0 => {
                zoom *= (1.0f32 + scroll_sens).powf(y);
            }
            _ => (),
        }
        if is_key_down(W) {
            pos += front * pan_sens;
        }
        if is_key_down(A) {
            pos -= right * pan_sens;
        }
        if is_key_down(S) {
            pos -= front * pan_sens;
        }
        if is_key_down(D) {
            pos += right * pan_sens;
        }
        if is_key_down(Q) {
            pos.y -= pan_sens;
        }
        if is_key_down(Z) {
            pos.y += pan_sens;
        }
        if is_key_down(KeyCode::Up) {
            editor_pos.y += pan_sens * zoom;
        }
        if is_key_down(KeyCode::Down) {
            editor_pos.y -= pan_sens * zoom;
        }
        if is_key_down(KeyCode::Right) {
            editor_pos.x += pan_sens * zoom;
        }
        if is_key_down(KeyCode::Left) {
            editor_pos.x -= pan_sens * zoom;
        }

        // INPUT SHIP EDITOR
        if let Some((x, y)) = mouse_grid {
            if is_mouse_button_down(MouseButton::Left) {
                grid[x][y] = match new_tile_type {
                    ShipTile::Input(_) => {
                        let new = ShipTile::Input(keybind_idx);
                        keybind_idx = (keybind_idx + 1) % KEY_BINDS.len();
                        new
                    }
                    _ => new_tile_type.clone(),
                };
                circuit_state_a = init_circuit(&grid);
            }
            if is_mouse_button_down(MouseButton::Right) {
                grid[x][y] = ShipTile::Ground;
                circuit_state_a = init_circuit(&grid);
            }
        }

        if is_key_pressed(O) {
            // SAVE FILE
            info!("Saving File");
            let mut file = File::create(ship_files[active_ship_file].as_path()).unwrap();
            file.write_all(SerRon::serialize_ron(&grid).as_bytes())
                .unwrap();

            // RELOAD TO ensure synced
            grid = hotload_ship(active_ship_file);
            circuit_state_a = init_circuit(&grid);
        }

        if is_key_pressed(L) {
            active_ship_file = (active_ship_file + 1) % ship_files.len();
            grid = hotload_ship(active_ship_file);
            circuit_state_a = init_circuit(&grid);
        }

        if is_key_pressed(Tab) {
            new_tile_type = tile_type_iter.next().unwrap();
        }
        if is_key_pressed(E) {
            selected_tile = mouse_grid;
        }
        if is_key_released(E) {
            selected_tile = None;
        }

        if let Some((x, y)) = selected_tile {
            let dir = Dir::from_vec2(mouse_world - vec2(x as f32, y as f32));

            match grid[x][y].clone() {
                ShipTile::Engine(_, bool) => grid[x][y] = ShipTile::Engine(dir, bool),
                _ => (),
            }
        }

        // UPDATE SHIP
        for (x, col) in grid.iter_mut().enumerate() {
            for (y, tile) in col.iter_mut().enumerate() {
                match tile {
                    ShipTile::Engine(dir, _) => {
                        let v = dir.v();
                        let xb = x as isize - v.x as isize;
                        let yb = y as isize - v.y as isize;
                        let bounds = circuit_state_a.len() as isize;
                        if xb > 0 && xb < bounds && yb > 0 && yb < bounds {
                            if circuit_state_a[xb as usize][yb as usize] == Circuit::Head {
                                circuit_state_a[xb as usize][yb as usize] = Circuit::Tail;
                                ship.f += v;
                                *tile = ShipTile::Engine(dir.clone(), true);
                            } else {
                                *tile = ShipTile::Engine(dir.clone(), false);
                            }
                        }
                    }

                    ShipTile::Input(key_idx) => {
                        if is_key_down(KEY_BINDS[*key_idx].0)
                            && circuit_state_a[x][y] == Circuit::Wire
                        {
                            circuit_state_a[x][y] = Circuit::Head;
                        }
                    }
                    _ => (),
                }
            }
        }

        // SIMULATE CIRCUIT
        if state_count == 0 {
            circuit_state_b = circuit_state_a.clone();
            for x in 0..circuit_state_b.len() {
                for y in 0..circuit_state_b[0].len() {
                    circuit_state_a[x][y] = match circuit_state_b[x][y] {
                        Circuit::None => Circuit::None,
                        Circuit::Wire => {
                            let mut count = 0;
                            for ix in -1..(2 as isize) {
                                let xn = x as isize + ix;
                                if xn > 0 && xn < circuit_state_b.len() as isize {
                                    let col = &circuit_state_b[xn as usize];
                                    for iy in -1..(2 as isize) {
                                        let yn = y as isize + iy;
                                        if yn > 0 && yn < circuit_state_b[0].len() as isize {
                                            if col[yn as usize] == Circuit::Head {
                                                count += 1;
                                            }
                                        }
                                    }
                                }
                            }
                            if count == 1 {
                                Circuit::Head
                            } else {
                                Circuit::Wire
                            }
                        }
                        Circuit::Head => Circuit::Tail,
                        Circuit::Tail => Circuit::Wire,
                    };
                }
            }
        }
        state_count = (state_count + 1) % 5;

        // DRAW EDITOR
        draw_grid(GRID_SLICES as u32, GRID_SCALE, BLACK, GRAY);
        if let Some(coord) = selected_tile {
            let sz = Vec3::splat(GRID_SCALE);
            draw_cube_wires(g3d(coord), sz, WHITE);
            draw_line_3d(g3d(coord), p3d(mouse_world), WHITE);
        }

        for (x, col) in grid.iter().enumerate() {
            for (y, tile) in col.iter().enumerate() {
                let p = g3d((x, y)) + GRID_OFFSET * UP;
                let sz = Vec3::splat(GRID_SCALE);
                let color = match circuit_state_b[x][y] {
                    Circuit::None => GRAY,
                    Circuit::Wire => YELLOW,
                    Circuit::Head => RED,
                    Circuit::Tail => BLUE,
                };

                match tile {
                    ShipTile::Ground => (),
                    ShipTile::Wall => draw_cube(p, sz, None, GRAY),
                    ShipTile::CenterOfMass => draw_cube(p, sz, None, RED),
                    ShipTile::Engine(dir, on) => {
                        draw_cube(p, sz / 2., None, ORANGE);
                        draw_sphere(
                            p + p3d(dir.v()) * GRID_SCALE / 4.,
                            GRID_SCALE / 4.,
                            None,
                            ORANGE,
                        );

                        if *on {
                            draw_sphere(
                                p + p3d(dir.v()) * GRID_SCALE,
                                GRID_SCALE / 8.,
                                None,
                                ORANGE,
                            );
                            draw_sphere(
                                p + p3d(dir.v()) * 2. * GRID_SCALE,
                                GRID_SCALE / 12.,
                                None,
                                ORANGE,
                            );
                            draw_sphere(
                                p + p3d(dir.v()) * 3. * GRID_SCALE,
                                GRID_SCALE / 16.,
                                None,
                                ORANGE,
                            );
                        }
                    }
                    ShipTile::Wire => {
                        draw_cube(p, sz / 4., None, color);
                    }
                    ShipTile::Input(_key_idx) => {
                        let _screen_p = to_screen(p, &editor_camera);
                        draw_cube(p, sz / 4., None, color);
                    }
                }
            }
        }

        // DRAW WORLD
        set_camera(&world_camera);
        draw_grid(GRID_SLICES as u32, GRID_SCALE, BLACK, GRAY);
        draw_scene_recursive(
            &ship_meshes,
            &ship_vox,
            0,
            None,
            IVec3::ZERO,
            Rotation::IDENTITY,
        );

        // DRAW UI
        set_default_camera();
        {
            draw_target(&editor_camera, vec2(screen_width() / 2., 0.));
            let mut anchor = Vec2::ZERO;
            draw_info_line(&mut anchor, format!("fps: {}", get_fps()).as_str());
            let ship_name = ship_files[active_ship_file].as_path().display();
            draw_info_line(&mut anchor, format!("ship file: {}", ship_name).as_str());
            draw_info_line(&mut anchor, format!("tile: {}", new_tile_type).as_str());

            next_frame().await
        }
    }
}
