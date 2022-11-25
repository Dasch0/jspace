use core::f32::consts::PI;
use core::ops::{Add, Sub};
use std::cmp::Ordering;

use rayon::prelude::*;
use macroquad::{prelude::KeyCode::*, prelude::*, rand::gen_range};
use bitflags::bitflags;

const DT: f32 = 0.1;
const G: f32 = 1.0;
const H: f32 = 10.0;
const ENGINE: f32 = 0.1;

const WORLD_SIZE: f32 = 50.0;
const SUN_SIZE: f32 = 10.0;

#[derive(Clone, Copy)]
struct Ship {
    pos: Vec2,
    vel: Vec2,
    dir: Vec2,
    capacity: f32,
    fuel: f32,
    pressure: f32,
    temp: f32,
    mass: f32,
    size: f32,
    color: Color,
}

#[derive(Clone, Copy)]
struct Planet {
    pos: Vec2,
    vel: Vec2,
    mass: f32,
    size: f32,
    color: Color,
}

fn gravity_force(pos: Vec2, cent: Planet) -> Vec2 {
    let g_v = cent.pos - pos;
    let a_g = G * cent.mass / g_v.length_squared();
    a_g * g_v.normalize()
}

/// Compute the speed required for circular orbit around a given gravitational center point
fn calc_orbital_speed(pos: Vec2, cent: Planet) -> Vec2 {
    let g_v = vec2(cent.pos.x - pos.x, cent.pos.y - pos.y);
    Vec2::from_angle(PI / 2.0).rotate(g_v.normalize()) * (G * cent.mass / g_v.length()).sqrt()
}

fn rand_initial_pos(cent: Planet) -> Vec2 {
    cent.pos
        + vec2(
            gen_range(cent.size, WORLD_SIZE),
            gen_range(cent.size, WORLD_SIZE),
        )
}

#[inline]
fn fcmp(a: f32, b: f32) -> std::cmp::Ordering {
    let diff = a - b;
    if diff.abs() < f32::EPSILON {
        std::cmp::Ordering::Equal
    } else {
        a.total_cmp(&b)
    }
}

/// A 2-dimensional vector.
#[derive(Clone, Copy, PartialEq, Debug)]
struct PVec2 {
    pub r: f32,
    pub t: f32,
}

#[inline]
fn pvec2(r: f32, t: f32) -> PVec2 {
    PVec2 {
        r,
        t,
    }
}

impl PVec2 {
    /// get polar coordinate from cartesian vec input
    #[inline]
    pub fn from_cart(pos: Vec2) -> PVec2 {
        let r = pos.length();
        let theta = pos.y.atan2(pos.x);
        pvec2(r, theta)
    }

    /// get polar velocity from cartesian velocity and position 
    /// @source: https://math.stackexchange.com/questions/2444965/relationship-between-cartesian-velocity-and-polar-velocity
    #[inline]
    pub fn from_cart_vel(pos: Vec2, vel: Vec2) -> PVec2 {
        let v_r = pos.dot(vel) / pos.length();
        let v_theta = (pos.x * vel.y - vel.x * pos.y) / pos.length_squared();
        pvec2(v_r, v_theta)
    }

    /// get cartesian velocity from polar velocity and position
    /// @source: https://math.stackexchange.com/questions/2444965/relationship-between-cartesian-velocity-and-polar-velocity
    pub fn to_cart_vel(self, pos: PVec2) -> Vec2 {
        let dx = self.r * f32::cos(pos.t) - pos.r * self.t * f32::sin(pos.t);
        let dy = self.r * f32::sin(pos.t) + pos.r * self.t * f32::cos(pos.t);
        vec2(dx, dy)
    }
}

impl Add for PVec2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            t: f32::sin(self.t + other.t).atan2(f32::cos(self.t + other.t)),
        }
    }
}

impl Sub for PVec2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            r: self.r - other.r, 
            t: f32::sin(self.t - other.t).atan2(f32::cos(self.t - other.t)),
        }
    }
}

// get polar velocity of cartesian velocity input
// orbital control equation, outputs a force vector to apply
// NOTE: currently assumes rhs orbit direction
fn calc_orbital_control(p: Vec2, v: Vec2, target_p: Vec2, target_v: Vec2, debug: bool) -> Vec2 {
    // project inputs into polar coords
    let p_pol = PVec2::from_cart(p);
    let target_p_pol = PVec2::from_cart(target_p);
    let v_pol = PVec2::from_cart_vel(p, v);
    let target_v_pol = PVec2::from_cart_vel(target_p, target_v);

    let delta_p = target_p_pol - p_pol;
    let delta_v = target_v_pol - v_pol;

    if debug {
        println!("p_pol: {:?}", p_pol);
        println!("target_p_pol: {:?}", target_p_pol);
        println!("v_pol: {:?}", v_pol);
        println!("target_v_pol: {:?}", target_v_pol);
        println!("delta_p: {:?}", delta_p);
        println!("delta_v: {:?}", delta_v);
    }

    // Simple PD controller, no integral term to keep stateless
    let control_f = pvec2(0.1 * delta_p.r + 0.9 * delta_v.r, 0.07 * delta_p.t + 0.9 * delta_v.t);
    control_f.to_cart_vel(p_pol)
}

#[macroquad::main("jspace")]
async fn main() {
    let mut ships: Vec<Ship> = Vec::new();
    let mut dust: Vec<Vec4> = Vec::new();
    let mut planets: Vec<Planet> = Vec::new();

    // SUN
    let sun = Planet {
            pos: vec2(0., 0.),
            vel: vec2(0., 0.),
            size: SUN_SIZE,
            mass: 10.0,
            color: YELLOW,
        };

    // PLAYER
    ships.push({
        let pos = rand_initial_pos(sun);
        Ship {
            pos,
            vel: calc_orbital_speed(pos, sun),
            dir: Vec2::X,
            capacity: 0.,
            fuel: 500.,
            pressure: 500.,
            temp: 100.0,
            mass: 1.0,
            size: 0.2,
            color: ORANGE,
        }
    });
    let player_idx = 0;

    // ENEMIES
    ships.extend((0..2).map(|i| {
        let pos = rand_initial_pos(sun);
        Ship {
            pos,
            vel: calc_orbital_speed(pos, sun),
            dir: Vec2::X,
            capacity: 100.,
            fuel: 1000.,
            pressure: 500.,
            temp: 100.0,
            mass: 1.0,
            size: 0.2,
            color: RED,
        }
    }));

    // ITEMS
    for _i in 0..1000 {
        let i_p = rand_initial_pos(sun);
        let i_v = calc_orbital_speed(i_p, sun);
        dust.push(vec4(i_p.x, i_p.y, i_v.x, i_v.y));
    }

    // PLANETS
    for _i in 0..4 {
        let pos = rand_initial_pos(sun);
        let vel = calc_orbital_speed(pos, sun);
        planets.push(Planet {
            pos,
            vel,
            size: 2.,
            mass: 1.0,
            color: GREEN,
        });
    }

    let mut input_mag: Vec2;
    let mut input_angle: f32 = 0.0;
    loop {
        clear_background(DARKGRAY);

        // INPUT
        input_mag = Vec2::ZERO;
        if is_key_down(W) {
            input_mag += Vec2::X
        }
        if is_key_down(A) {
            input_angle -= PI / 128.0
        }
        if is_key_down(D) {
            input_angle += PI / 128.0
        }

        let input_dir = Vec2::from_angle(input_angle).rotate(Vec2::X * 2.0);
        let input_force = Vec2::from_angle(input_angle).rotate(input_mag);

        // SHIP CONTROL
        // ships[player_idx].dir = input_dir;
        // ships[player_idx].vel += input_force.normalize_or_zero() * ENGINE * DT;

        let player_p = ships[player_idx].pos;
        let player_v = ships[player_idx].vel;
        ships.par_iter_mut().enumerate().for_each(|(idx, ship)| {
            let target_p: Vec2;
            let target_v: Vec2;
            if ship.capacity >= 500. {
                let target_idx = idx % planets.len();
                target_p = planets[target_idx].pos;
                target_v = planets[target_idx].vel;
                if idx == 0 {
                    println!("target: planet at: {:?}", target_p);
                }
            } else {
                let target_idx = idx % dust.len();
                target_p = dust[target_idx].xy();
                target_v = dust[target_idx].zw();
               
                if idx == 0 {
                    println!("target: asteroid at: {:?}", target_p);
                }
            }
            let control_force = calc_orbital_control(ship.pos, ship.vel, target_p, target_v, (idx == 0));
            if control_force == Vec2::ZERO { panic!("control force was zero"); }
            ship.dir = control_force.normalize();
            ship.vel += control_force * ENGINE * DT;
        });

        // STATIC ENT UPDATE
        planets.par_iter_mut().for_each(|planet| {
            let f_g = gravity_force(planet.pos, sun);
            planet.vel += f_g * DT;
            planet.pos += planet.vel * DT;
        });

        dust.par_iter_mut().for_each(|obj| {
            let f_g = gravity_force(obj.xy(), sun);
            obj.z += f_g.x * DT;
            obj.w += f_g.y * DT;
            obj.x += obj.z * DT;
            obj.y += obj.w * DT;
        });

        // DYNAMIC ENT UPDATE
        ships.par_iter_mut().for_each(|ship| {
            let f_g = gravity_force(ship.pos, sun);
            ship.vel += f_g * DT;

            // PLANET COLLISION
            let candidate_pos = ship.pos + ship.vel * DT;
            for planet in planets.iter() {
                let dist = candidate_pos.distance(planet.pos);
                if dist <= planet.size / 2. {
                    ship.fuel = 500.;
                    ship.capacity = 0.;
                    ship.vel = planet.vel;
                }
            }

            // SUN COLLISION
            {
                let dist = candidate_pos.distance(sun.pos);
                if dist <= sun.size / 2. {
                    ship.pos = rand_initial_pos(sun);
                    ship.vel = calc_orbital_speed(ship.pos, sun);
                }
            }

            ship.pos += ship.vel * DT;

            // PICKUP
            for obj in dust.iter() {
                let dist = candidate_pos.distance(obj.xy());
                if dist <= 0.3 {
                    ship.capacity += 1.0;
                }
            }
        });

        // DRAW WORLD
        set_camera(&{
            let pos = ships[player_idx].pos;
            Camera3D {
                position: vec3(
                    pos.x - input_dir.x * 5.0,
                    H + 5.,
                    pos.y - input_dir.y * 5.0,
                ),
                up: vec3(0., 1., 0.),
                target: vec3(pos.x + input_dir.x, H + 1.0, pos.y + input_dir.y),
                ..Default::default()
            }
        });

        draw_grid(200, 1., BLACK, GRAY);

        // SUN
        draw_cube(vec3(sun.pos.x, H, sun.pos.y), Vec3::splat(sun.size), None, sun.color);

        for ship in ships.iter() {
            let pos = vec3(ship.pos.x, H, ship.pos.y);
            draw_line_3d(pos,
                pos + vec3(ship.dir.x, 0., ship.dir.y),
                GREEN,
            );

            draw_cube(pos, Vec3::splat(ship.size), None, ship.color);
        }

        for planet in planets.iter() {
            draw_cube(
                vec3(planet.pos.x, H, planet.pos.y),
                Vec3::splat(planet.size),
                None,
                planet.color,
            );
        }

        for obj in dust.iter() {
            draw_cube(vec3(obj.x, H, obj.y), Vec3::splat(0.1), None, WHITE);
        }

        // DRAW UI
        set_default_camera();
        draw_text(
            format!("FPS: {}", get_fps()).as_str(),
            20.0,
            20.0,
            20.0,
            GREEN,
        );

        draw_rectangle(
            screen_width() - 100.,
            screen_height() - 50. - ships[player_idx].capacity,
            5.,
            ships[player_idx].capacity,
            WHITE,
        );
        draw_rectangle(
            screen_width() - 80.,
            screen_height() - 50. - ships[player_idx].fuel,
            5.,
            ships[player_idx].fuel,
            ORANGE,
        );

        next_frame().await
    }
}
