use log::{debug, info, trace};
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

// ------------------------------------------------------------------------------------------------

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

#[derive(Clone, SerRon, DeRon, Default)]
struct Circuit {
    dir: Dir,
    v: f32,
    i: f32,
    r: f32,
    q: f32,
    c: f32,
    l: f32,
}

#[derive(
    Default,
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
    Engine(Dir),
    Pipe(Circuit),
    Valve(Circuit),
    Actuator(Circuit),
    Tank(Circuit),
    Pump(Circuit),
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

#[macroquad::main("jspace")]
async fn main() {
    env_logger::init();

    // CONFIGURATIONS
    let scroll_sens = 0.001;
    let pan_sens: f32 = 0.1;

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

    println!("{:?}", ship_files);
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
                                                               // Ground

    // WORLD ELEMENTS
    let mut grid = hotload_ship(0);
    let ship = Ship::default();
    let map = Vec::<Terrain>::new();

    // INPUT ELEMENTS
    let mut pos = vec2(grid.len() as f32 / 2., grid.len() as f32 / 2.);
    let mut zoom = 1.0;
    let mut active_ship_file: usize = 0;
    let mut new_tile_type: ShipTile = tile_type_iter.next().unwrap();
    let mut selected_tile: Option<(usize, usize)> = None;

    loop {
        // PREPARE FRAME
        clear_background(DARKGRAY);
        let camera = Camera3D {
            position: vec3(pos.x, -15. * zoom, pos.y),
            up: UP,
            target: vec3(pos.x, 0., pos.y + 1.0),
            ..Default::default()
        };
        set_camera(&camera);

        // USER INPUT
        let mouse_world: Vec2 = {
            // SOURCE: https://antongerdelan.net/opengl/raycasting.html
            let clip = (mouse_position_local() * vec2(1.0, -1.0))
                .extend(-1.0)
                .extend(1.0);
            let eye = (camera.proj().inverse() * clip)
                .xy()
                .extend(-1.0)
                .extend(0.0);
            let ray = (camera.view().inverse() * eye).xyz().normalize();

            let distance_to_ground = -camera.position.dot(UP) / (ray.dot(UP));
            (camera.position + ray * distance_to_ground).xz()
        };
        d!(mouse_world);
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
        d!(mouse_grid);
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
                grid[x][y] = new_tile_type.clone();
            }
            if is_mouse_button_down(MouseButton::Right) {
                grid[x][y] = ShipTile::Ground;
            }
        }

        if is_key_pressed(O) {
            // SAVE FILE
            info!("Saving File");
            let mut file = File::create(ship_files[active_ship_file].as_path()).unwrap();
            file.write_all(SerRon::serialize_ron(&grid).as_bytes())
                .unwrap();
        }

        if is_key_pressed(L) {
            active_ship_file = (active_ship_file + 1) % ship_files.len();
            grid = hotload_ship(active_ship_file);
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
            t!(dir);

            match grid[x][y].clone() {
                ShipTile::Engine(_) => grid[x][y] = ShipTile::Engine(dir),
                ShipTile::Pipe(circuit) => {
                    grid[x][y] = {
                        let mut new_circuit = circuit.clone();
                        new_circuit.dir = dir;
                        ShipTile::Pipe(new_circuit)
                    }
                }
                ShipTile::Valve(circuit) => {
                    grid[x][y] = {
                        let mut new_circuit = circuit.clone();
                        new_circuit.dir = dir;
                        ShipTile::Valve(new_circuit)
                    }
                }
                ShipTile::Actuator(circuit) => {
                    grid[x][y] = {
                        let mut new_circuit = circuit.clone();
                        new_circuit.dir = dir;
                        ShipTile::Actuator(new_circuit)
                    }
                }
                ShipTile::Tank(circuit) => {
                    grid[x][y] = {
                        let mut new_circuit = circuit.clone();
                        new_circuit.dir = dir;
                        ShipTile::Actuator(new_circuit)
                    }
                }
                _ => (),
            }
        }

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

                match tile {
                    ShipTile::Ground => (),
                    ShipTile::Wall => draw_cube(p, sz, None, GRAY),
                    ShipTile::CenterOfMass => draw_cube(p, sz, None, RED),
                    ShipTile::Engine(dir) => {
                        draw_cube(p, sz / 2., None, ORANGE);
                        draw_sphere(
                            p + p3d(dir.v()) * GRID_SCALE / 4.,
                            GRID_SCALE / 4.,
                            None,
                            ORANGE,
                        );
                    }
                    ShipTile::Pipe(circuit) => {
                        let dir = p3d(circuit.dir.v() * GRID_SCALE / 4.);
                        draw_cube(p, sz / 8., None, BLUE);
                        draw_cube(p + dir, sz / 8., None, BLUE);
                        draw_cube(p - dir, sz / 8., None, BLUE);
                    }
                    ShipTile::Valve(circuit) => {
                        let dir = p3d(circuit.dir.v() * GRID_SCALE / 4.);
                        draw_cube(p, sz / 4., None, BLUE);
                        draw_cube(p + dir, sz / 8., None, BLUE);
                        draw_cube(p - dir, sz / 8., None, BLUE);
                    }
                    ShipTile::Actuator(circuit) => {
                        let color = Color {
                            r: circuit.v,
                            g: circuit.v,
                            b: 1.0,
                            a: 1.0,
                        };
                        draw_cube(p, sz / 2., None, color);
                    }
                    ShipTile::Tank(circuit) => {
                        let fill = vec3(1.0, circuit.q / circuit.c * circuit.v, 1.0);
                        draw_cube(p, sz * fill / 2., None, BLUE);
                        draw_cube_wires(p, sz / 2., BLUE);
                    }
                    ShipTile::Pump(circuit) => {
                        let dir = p3d(circuit.dir.v() * GRID_SCALE / 4.);
                        draw_cube(p, sz / 8., None, BLUE);
                        draw_cube(p + dir, sz / 4., None, BLUE);
                        draw_cube(p - dir, sz / 8., None, BLUE);
                    }
                }
            }
        }

        // DRAW UI
        set_default_camera();
        {
            let mut anchor = Vec2::ZERO;
            draw_info_line(&mut anchor, format!("fps: {}", get_fps()).as_str());
            let ship_name = ship_files[active_ship_file].as_path().display();
            draw_info_line(&mut anchor, format!("ship file: {}", ship_name).as_str());
            draw_info_line(&mut anchor, format!("tile: {}", new_tile_type).as_str());

            next_frame().await
        }
    }
}
