use core::f32::consts::PI;
use core::ops::{Add, Sub};
use crossbeam_channel::unbounded;
use macroquad::{prelude::KeyCode::*, prelude::*, rand::gen_range};
use rayon::prelude::*;

const DT: f32 = 0.1;
const G: f32 = 1.0;
const H: f32 = 10.0;
const ENGINE: f32 = 0.2;

const WORLD_SIZE: f32 = 50.0;
const SUN_SIZE: f32 = 10.0;

struct Controller {
    target_pos: Vec2,
    target_vel: Vec2,
    aim_pos: Vec2,
    aim_vel: Vec2,
    state: u8,
}

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
    cooldown: u8,
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

struct Projectile {
    start: Vec2,
    pos: Vec2,
    target: Vec2,
    state: u8,
    timer: u8,
    lifespan: u8,
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
    PVec2 { r, t }
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
    let control_f = pvec2(
        0.1 * delta_p.r + 0.99 * delta_v.r - 2.0 * delta_v.t,
        0.02 * delta_p.t.atan().atan() + 0.50 * delta_v.t,
    );
    control_f.to_cart_vel(p_pol)
}

#[macroquad::main("jspace")]
async fn main() {
    let mut countdown = 5000;
    let mut ships: Vec<Ship> = Vec::new();
    let mut controllers: Vec<Controller> = Vec::new();
    let mut dust: Vec<Vec4> = Vec::new();
    let mut planets: Vec<Planet> = Vec::new();
    let (new_projectile_tx, new_projectile_rx) = unbounded::<Projectile>();
    let mut projectiles: Vec<Projectile> = Vec::new();

    // SUN
    let sun = Planet {
        pos: vec2(0., 0.),
        vel: vec2(0., 0.),
        size: SUN_SIZE,
        mass: 10.0,
        color: YELLOW,
    };

    // PLAYER
    let mut player_idx = 0;

    // SHIPS
    for _i in 0..100 {
        let pos = rand_initial_pos(sun);
        ships.push(Ship {
            pos,
            vel: calc_orbital_speed(pos, sun),
            dir: Vec2::X,
            capacity: 100.,
            fuel: 2000.,
            pressure: 500.,
            temp: 100.0,
            mass: 1.0,
            size: 0.2,
            color: match rand::rand() >= u32::MAX / 2 {
                true => MAROON,
                false => SKYBLUE,
            },
            cooldown: 0,
        });
        controllers.push(Controller {
            state: 0,
            target_pos: Vec2::ZERO,
            target_vel: Vec2::ZERO,
            aim_pos: Vec2::ZERO,
            aim_vel: Vec2::ZERO,
        });
    }
    assert!(controllers.len() == ships.len());

    // ITEMS
    for _i in 0..1000 {
        let i_p = rand_initial_pos(sun);
        let i_v = calc_orbital_speed(i_p, sun);
        dust.push(vec4(i_p.x, i_p.y, i_v.x, i_v.y));
    }

    // PLANETS
    for _i in 0..2 {
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
        if countdown > 0 {
            countdown -= 1;
        }
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
        if is_key_pressed(Tab) {
            player_idx = (player_idx + 1) % ships.len();
        }

        let input_dir = Vec2::from_angle(input_angle).rotate(Vec2::X * 2.0);
        let input_force = Vec2::from_angle(input_angle).rotate(input_mag);

        // KILL ENTITIES
        projectiles = projectiles
            .into_iter()
            .filter(|x| x.state & 0b1 > 0)
            .collect();
        controllers
            .par_iter_mut()
            .zip(ships.par_iter_mut())
            .for_each(|(controller, ship)| {
                let team_roll = rand::rand() >= u32::MAX / 2;
                if ship.pressure <= 0. {
                    let pos = planets[team_roll as usize].pos
                        + Vec2::splat(planets[team_roll as usize].size * 1.1);
                    *controller = Controller {
                        state: 0,
                        target_pos: Vec2::ZERO,
                        target_vel: Vec2::ZERO,
                        aim_pos: Vec2::ZERO,
                        aim_vel: Vec2::ZERO,
                    };
                    *ship = Ship {
                        pos,
                        vel: calc_orbital_speed(pos, sun),
                        dir: Vec2::X,
                        capacity: 100.,
                        fuel: 2000.,
                        pressure: 500.,
                        temp: 100.0,
                        mass: 1.0,
                        size: 0.2,
                        color: match team_roll {
                            true => MAROON,
                            false => SKYBLUE,
                        },
                        cooldown: 0,
                    };
                }
            });

        // NEW ENTITIES
        while let Ok(p) = new_projectile_rx.try_recv() {
            projectiles.push(p);
        }

        // SHIP DECISION
        ships[player_idx].dir = input_dir;
        ships[player_idx].vel += input_force.normalize_or_zero() * ENGINE * DT;

        controllers
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, controller)| {
                let my_ship = ships[idx];
                let mut target: Option<Ship> = None;
                let mut min_dist = f32::MAX;

                // evaluate potential targets
                for ship in ships[0..idx].iter().chain(ships[idx..].iter().skip(1)) {
                    let dist = my_ship.pos.distance(ship.pos);
                    if dist < 5.0 && my_ship.color != ship.color {
                        target = match target {
                            None => {
                                min_dist = dist;
                                Some(*ship)
                            }
                            Some(old_target) => {
                                if min_dist < dist {
                                    Some(old_target)
                                } else {
                                    min_dist = dist;
                                    Some(*ship)
                                }
                            }
                        }
                    }
                }

                if let Some(t) = target {
                    if countdown == 0 {
                        controller.state |= 0b10;
                    }
                    controller.aim_pos = t.pos;
                    controller.aim_vel = t.vel;
                } else {
                    controller.state &= !0b10;
                }

                if my_ship.capacity > 500. {
                    controller.state = 0b1;
                    let target_idx = if my_ship.color == MAROON { 1 } else { 0 };
                    controller.target_pos = planets[target_idx].pos;
                    controller.target_vel = planets[target_idx].vel;
                } else {
                    controller.state &= !0b1;
                    let target_idx = idx % dust.len();
                    controller.target_pos = dust[target_idx].xy();
                    controller.target_vel = dust[target_idx].zw();
                }
            });

        // SHIP ACTION
        ships
            .par_iter_mut()
            .zip(&controllers)
            .enumerate()
            .for_each(|(idx, (ship, controller))| {
                if ship.cooldown > 0 {
                    ship.cooldown -= 1;
                }

                if (controller.state & 0b10) > 0 && ship.cooldown == 0 {
                    // attack!
                    new_projectile_tx
                        .send({
                            let target = controller.aim_pos + controller.aim_vel * DT * 60.;
                            let dir = (target - ship.pos).normalize_or_zero();
                            let pos = ship.pos + dir * (ship.size * 2.);
                            Projectile {
                                start: pos,
                                pos,
                                target,
                                state: 0b1,
                                timer: 60,
                                lifespan: 60,
                            }
                        })
                        .expect("failed to access create entity queue");

                    ship.cooldown = 5;
                }

                let control_force = calc_orbital_control(
                    ship.pos,
                    ship.vel,
                    controller.target_pos,
                    controller.target_vel,
                    false,
                );
                if control_force == Vec2::ZERO {
                    println!("WARN: control force was zero!");
                    ship.dir = ship.dir;
                    ship.vel = ship.vel;
                } else {
                    let control_magnitude = control_force.length();
                    if ship.fuel > 0. {
                        ship.fuel -= control_magnitude;
                        ship.dir = control_force.normalize();
                        ship.vel += ship.dir * f32::min(control_magnitude, ENGINE) / ship.mass * DT;
                    }
                }
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
        ships.par_iter_mut().enumerate().for_each(|(idx, ship)| {
            let f_g = gravity_force(ship.pos, sun);
            ship.vel += f_g * DT;
            let candidate_pos = ship.pos + ship.vel * DT;

            // PROJECTILE COLLISION
            for projectile in projectiles.iter() {
                let dist = candidate_pos.distance(projectile.pos);
                if dist <= ship.size {
                    ship.mass += 0.01;
                    ship.pressure -= 10.0;
                    ship.vel +=
                        (projectile.target - projectile.start).normalize_or_zero() * ENGINE * DT;
                }
            }

            // PLANET COLLISION
            for planet in planets.iter() {
                let dist = candidate_pos.distance(planet.pos);
                if dist <= planet.size {
                    ship.fuel = 2000.;
                    ship.capacity = 0.;
                    ship.vel = planet.vel;
                }
            }

            // SUN COLLISION
            {
                let dist = candidate_pos.distance(sun.pos);
                if dist <= sun.size {
                    ship.pressure = -0.1;
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

        projectiles.par_iter_mut().for_each(|projectile| {
            // SHIP COLLISION
            for ship in ships.iter() {
                let dist = projectile.pos.distance(ship.pos);
                if dist <= ship.size {
                    projectile.state &= !0b1;
                }
            }

            // PLANET COLLISION
            for planet in planets.iter() {
                let dist = projectile.pos.distance(planet.pos);
                if dist <= planet.size {
                    projectile.state &= !0b1;
                }
            }

            // SUN COLLISION
            {
                let dist = projectile.pos.distance(sun.pos);
                if dist <= sun.size {
                    projectile.state &= !0b1;
                }
            }

            projectile.pos +=
                (projectile.target - projectile.start) / (projectile.lifespan - 1) as f32;

            if projectile.timer > 0 {
                projectile.timer -= 1;
            } else {
                projectile.state &= !0b1;
            }
        });

        // DRAW WORLD
        let camera = {
            let pos = ships[player_idx].pos;
            Camera3D {
                position: vec3(pos.x - input_dir.x * 5.0, H + 5., pos.y - input_dir.y * 5.0),
                up: vec3(0., 1., 0.),
                target: vec3(pos.x + input_dir.x, H + 1.0, pos.y + input_dir.y),
                ..Default::default()
            }
        };
        set_camera(&camera);
        draw_grid(200, 1., BLACK, GRAY);

        // SUN
        draw_sphere(vec3(sun.pos.x, H, sun.pos.y), sun.size, None, sun.color);

        {
            // CURRENT PLAYER DATA
            let pos = vec3(ships[player_idx].pos.x, H, ships[player_idx].pos.y);
            let target_pos = vec3(
                controllers[player_idx].target_pos.x,
                H,
                controllers[player_idx].target_pos.y,
            );
            draw_line_3d(pos, target_pos, WHITE);
        }

        for (ship, controller) in ships.iter().zip(controllers.iter()) {
            let pos = vec3(ship.pos.x, H, ship.pos.y);
            let aim_pos = vec3(controller.aim_pos.x, H, controller.aim_pos.y);
            draw_line_3d(pos, pos + vec3(ship.dir.x, 0., ship.dir.y), GREEN);
            if (controller.state & 0b10) > 0 {
                draw_line_3d(pos, aim_pos, BLUE);
            }
            draw_cube(pos, Vec3::splat(ship.size), None, ship.color);

            let o_2d = Vec2::from_angle(input_angle).rotate(vec2(0.0, 0.15));
            let o = vec3(o_2d.x, 0., o_2d.y);
            draw_cube(
                pos + o + vec3(0., ship.fuel / 4000., 0.),
                vec3(0.04, ship.fuel / 2000., 0.04),
                None,
                ORANGE,
            );
            draw_cube(
                pos + 2. * o + vec3(0., ship.capacity / 2000., 0.),
                vec3(0.04, ship.capacity / 1000., 0.04),
                None,
                WHITE,
            );
            draw_cube(
                pos + 3. * o + vec3(0., ship.pressure / 2000., 0.),
                vec3(0.04, ship.pressure / 1000., 0.04),
                None,
                BLUE,
            );
            draw_cube(
                pos + 4. * o + vec3(0., ship.mass / 2000., 0.),
                vec3(0.04, ship.mass / 1000., 0.04),
                None,
                RED,
            );
        }

        for planet in planets.iter() {
            draw_sphere(
                vec3(planet.pos.x, H, planet.pos.y),
                planet.size,
                None,
                planet.color,
            );
        }

        for obj in dust.iter() {
            draw_cube(vec3(obj.x, H, obj.y), Vec3::splat(0.1), None, WHITE);
        }

        for obj in projectiles.iter() {
            draw_cube(vec3(obj.pos.x, H, obj.pos.y), Vec3::splat(0.05), None, BLUE);
            draw_line_3d(
                vec3(obj.pos.x, H, obj.pos.y),
                vec3(obj.target.x, H, obj.target.y),
                RED,
            );
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

        draw_rectangle(
            screen_width() - 60.,
            screen_height() - 50. - ships[player_idx].pressure,
            5.,
            ships[player_idx].pressure,
            BLUE,
        );

        draw_rectangle(
            screen_width() - 40.,
            screen_height() - 50. - ships[player_idx].mass * 10.,
            5.,
            ships[player_idx].mass * 10.,
            RED,
        );

        next_frame().await
    }
}
