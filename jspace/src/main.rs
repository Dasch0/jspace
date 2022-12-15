use log::{info, trace};
use macroquad::input::KeyCode::*;
use macroquad::prelude::*;
use nanoserde::{DeRon, SerRon};
use std::{
    f32::consts::PI,
    fs::{self, File},
    io::prelude::*,
};
use strum::IntoEnumIterator;
use strum_macros::{Display, EnumCount, EnumIter, EnumString, EnumVariantNames, IntoStaticStr};
use rayon::prelude::*;

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
    draw_texture_ex(camera.render_target.unwrap().texture, p.x, p.y, WHITE, params);
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
    Tail
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
    t!(p);
    let coord = camera.matrix().project_point3(p).xy() * vec2(1.0, -1.0);
    t!(coord);
    let window = vec2(screen_width(), screen_height());
    let res = coord * window / 2. + window / 2.;
    t!(res);
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

#[macroquad::main(window_conf)]
async fn main() {
    env_logger::init();

    // CONFIGURATIONS
    let scroll_sens = 0.1;
    let pan_sens: f32 = 0.1;

    // MODELS
    let ship_vox = dot_vox::load("data/models/cargo-spaceship-by-fps-agency.vox").unwrap();
    i!(ship_vox);

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
    let mut pos = vec2(grid.len() as f32 / 2., grid.len() as f32 / 2.);
    let mut zoom = 1.0;
    let mut active_ship_file: usize = 0;
    let mut new_tile_type: ShipTile = tile_type_iter.next().unwrap();
    let mut selected_tile: Option<(usize, usize)> = None;

    let mut keybind_idx = 0;
    let mut state_count = 0;


    loop {
        // PREPARE FRAME
        clear_background(DARKGRAY);
        let editor_camera = Camera3D {
            position: vec3(pos.x, -15. * zoom, pos.y),
            up: UP,
            target: vec3(pos.x, 0., pos.y + 1.0),
            render_target: Some(render_target(screen_width() as u32 / 2, screen_height() as u32 / 2)),
            ..Default::default()
        };
        let world_camera = Camera3D {
            position: vec3(pos.x, -50. * zoom, pos.y),
            up: UP,
            target: vec3(pos.x + 5., 0., pos.y + 5.),
            render_target: Some(render_target(screen_width() as u32, screen_height() as u32)),
            ..Default::default()
        };

        set_camera(&editor_camera);

        // USER INPUT
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
            pos.y += pan_sens * zoom;
        }
        if is_key_down(A) {
            pos.x -= pan_sens * zoom;
        }
        if is_key_down(S) {
            pos.y -= pan_sens * zoom;
        }
        if is_key_down(D) {
            pos.x += pan_sens * zoom;
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
                    },

                    ShipTile::Input(key_idx) => if is_key_down(KEY_BINDS[*key_idx].0) && circuit_state_a[x][y] == Circuit::Wire {
                        circuit_state_a[x][y] = Circuit::Head;
                    },
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
                            draw_sphere(p + p3d(dir.v()) * GRID_SCALE, GRID_SCALE / 8., None, ORANGE);
                            draw_sphere(p + p3d(dir.v()) * 2. * GRID_SCALE, GRID_SCALE / 12., None, ORANGE);
                            draw_sphere(p + p3d(dir.v()) * 3. * GRID_SCALE, GRID_SCALE / 16., None, ORANGE);
                        }
                    }
                    ShipTile::Wire => {
                        draw_cube(p, sz / 4., None, color);
                    }
                    ShipTile::Input(key_idx) => {
                        let screen_p = to_screen(p, &editor_camera);
                        draw_cube(p, sz / 4., None, color);
                    }
                }
            }
        }

        // DRAW WORLD
        set_camera(&world_camera);
        let gl = unsafe { get_internal_gl().quad_gl };
        draw_grid(GRID_SLICES as u32, GRID_SCALE, BLACK, GRAY);
        let scale = vec3(0.1, -0.1, 0.1);
        let rotation = Quat::from_axis_angle(UP, 90.);
        let translation = vec3(100., 0., 100.);
        let ship_model_transform = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        gl.push_model_matrix(ship_model_transform);
        let mut t = 0.0;
        for model in ship_vox.models.iter() {
            let component_transform = Mat4::from_translation(vec3(t,0.,0.));
            gl.push_model_matrix(component_transform);
            for voxel in model.voxels.iter() {
                let p = vec3(voxel.x as f32, voxel.y as f32, voxel.z as f32);
                let sz = vec3(1.1, 1.1, 1.1);
                let c = &ship_vox.palette[voxel.i as usize];
                draw_cube(p, sz, None, Color::from_rgba(c.r, c.g, c.b, 254));
            }
            t += 50.;
            gl.pop_model_matrix();
        }
        gl.pop_model_matrix();

        // DRAW UI
        set_default_camera();
        {
            draw_target(&editor_camera, vec2(screen_width() / 2., 0.)); 
            draw_target(&world_camera, vec2(0., 0.)); 
            let mut anchor = Vec2::ZERO;
            draw_info_line(&mut anchor, format!("fps: {}", get_fps()).as_str());
            let ship_name = ship_files[active_ship_file].as_path().display();
            draw_info_line(&mut anchor, format!("ship file: {}", ship_name).as_str());
            draw_info_line(&mut anchor, format!("tile: {}", new_tile_type).as_str());

            next_frame().await
        }
    }
}
