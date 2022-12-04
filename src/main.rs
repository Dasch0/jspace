use core::f32::consts::PI;
use core::ops::{Add, Index, IndexMut, Sub};
use crossbeam_channel::unbounded;
use macroquad::{prelude::KeyCode::*, prelude::*, rand::gen_range};
use rayon::prelude::*;
use std::collections::BinaryHeap;

const DT: f32 = 0.1;
const G: f32 = 0.1;
const H: f32 = 10.0;
const ENGINE: f32 = 0.1;

const WORLD_SIZE: f32 = 50.0;

type Idx = u32;

#[repr(u32)]
#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub enum EID {
    Ship = 0,
    Planet = 1,
    Projectile = 2,
    Asteroid = 3,
    // NOTE: update EntityId::n impl! if adding more enums
}

impl EID {
    /// number of elements in the EntityId list
    pub const fn n() -> usize {
        4
    }

    pub const fn from(n: usize) -> EID {
        unsafe { std::mem::transmute(n as u32) }
    }
}
pub struct EntityList<T> {
    pub lists: [Vec<T>; EID::n()],
}

impl<T> EntityList<T> {
    pub fn new() -> EntityList<T> {
        EntityList {
            lists: [0; EID::n()].map(|_| Vec::<T>::new()),
        }
    }
    pub fn put(&mut self, e: EID, item: T) -> Idx {
        let idx = self.lists[e as usize].len();
        self.lists[e as usize].push(item);
        idx as u32
    }

    /// swap removes an item, and returns a ref of the MOVED item, not the removed one
    /// panics if index is invalid or collection is empty
    pub fn del(&mut self, e: EID, idx: u32) -> &T {
        self.lists[e as usize].swap_remove(idx as usize);
        &self.lists[e as usize][idx as usize]
    }
}

impl<T> Index<EID> for EntityList<T> {
    type Output = Vec<T>;

    fn index(&self, eid: EID) -> &Self::Output {
        &self.lists[eid as usize]
    }
}

impl<T> IndexMut<EID> for EntityList<T> {
    fn index_mut(&mut self, eid: EID) -> &mut Self::Output {
        &mut self.lists[eid as usize]
    }
}

pub struct EntityReceiver<T> {
    pub rx: [crossbeam_channel::Receiver<T>; EID::n()],
}

pub struct EntitySender<T> {
    pub tx: [crossbeam_channel::Sender<T>; EID::n()],
}

impl<T> Index<EID> for EntitySender<T> {
    type Output = crossbeam_channel::Sender<T>;

    fn index(&self, eid: EID) -> &Self::Output {
        &self.tx[eid as usize]
    }
}

impl<T> Index<EID> for EntityReceiver<T> {
    type Output = crossbeam_channel::Receiver<T>;

    fn index(&self, eid: EID) -> &Self::Output {
        &self.rx[eid as usize]
    }
}

pub fn entity_queue<T>() -> (EntitySender<T>, EntityReceiver<T>) {
    // FIXME: kind of a gross way to instantiate two lists of sender/receiver, but this is a first
    // attempt to dance around array instantiation limits
    let arr = [0; EID::n()];
    let queues = arr.map(|_| unbounded::<T>());
    (
        EntitySender {
            tx: queues.clone().map(|q| q.0),
        },
        EntityReceiver {
            rx: queues.map(|q| q.1),
        },
    )
}

pub trait PutDel {
    type Item;
    fn put(&mut self, item: Self::Item) -> Idx;
    fn del(&mut self, idx: Idx) -> &Self::Item;
}

impl<T> PutDel for Vec<T> {
    type Item = T;

    fn put(&mut self, item: Self::Item) -> u32 {
        let idx = self.len();
        self.push(item);
        idx as u32
    }

    /// swap removes an item, and returns a ref of the MOVED item, not the removed one
    /// panics if index is invalid or collection is empty
    fn del(&mut self, idx: u32) -> &T {
        self.swap_remove(idx as usize);
        &self[idx as usize]
    }
}

pub enum Entity {
    None,
    Ship { me: Idx, phys: Idx },
    Planet { me: Idx, phys: Idx },
    Projectile { me: Idx, phys: Idx },
    Asteroid { me: Idx, phys: Idx },
}

pub struct Controller {
    pub target_pos: Vec2,
    pub target_vel: Vec2,
    pub aim_pos: Vec2,
    pub aim_vel: Vec2,
    pub state: u8,
}

#[derive(Clone, Copy)]
pub struct Ship {
    pub capacity: f32,
    pub fuel: f32,
    pub pressure: f32,
    pub temp: f32,
    pub mass: f32,
    pub cooldown: u8,
    pub size: f32,
    pub color: Color,
}

#[derive(Clone, Copy)]
pub struct Planet {
    pub color: Color,
}

pub struct Asteroid {
    pub integrity: f32,
    pub color: Color,
}

pub struct Projectile {
    pub state: u8,
    pub timer: u8,
    pub lifespan: u8,
}

#[derive(Clone, Copy)]
pub struct PhysicsEntity {
    pub p: Vec2,
    pub v: Vec2,
    pub sz: f32,
    pub m: f32,
}

pub fn sun() -> (PhysicsEntity, Planet) {
    (
        PhysicsEntity {
            p: vec2(0., 0.),
            v: vec2(0., 0.),
            sz: 10.,
            m: 100.,
        },
        Planet { color: YELLOW },
    )
}

pub fn rand_planet(sun: PhysicsEntity) -> (PhysicsEntity, Planet) {
    let p = rand_initial_pos(sun);
    let v = calc_orbital_speed(p, sun);
    (
        PhysicsEntity {
            p,
            v,
            sz: 2.,
            m: 10.,
        },
        Planet { color: GREEN },
    )
}

pub fn rand_asteroid(sun: PhysicsEntity) -> (PhysicsEntity, Asteroid) {
    let p = rand_initial_pos(sun);
    let v = calc_orbital_speed(p, sun);
    (
        PhysicsEntity {
            p,
            v,
            sz: 0.2,
            m: 0.2,
        },
        Asteroid {
            integrity: 100.,
            color: LIGHTGRAY,
        },
    )
}

pub fn rand_drone(sun: PhysicsEntity) -> (PhysicsEntity, Ship) {
    let p = rand_initial_pos(sun);
    let v = calc_orbital_speed(p, sun);
    (
        PhysicsEntity {
            p,
            v,
            sz: 0.1,
            m: 1.0,
        },
        Ship {
            capacity: 100.,
            fuel: 2000.,
            pressure: 500.,
            temp: 100.0,
            mass: 1.0,
            size: 0.2,
            color: SKYBLUE,
            cooldown: 0,
        },
    )
}

fn gravity_force(pos: Vec2, cent: PhysicsEntity) -> Vec2 {
    let g_v = cent.p - pos;
    let a_g = G * cent.m / g_v.length_squared();
    a_g * g_v.normalize_or_zero()
}

/// Compute the speed required for circular orbit around a given gravitational center point
fn calc_orbital_speed(pos: Vec2, cent: PhysicsEntity) -> Vec2 {
    let g_v = vec2(cent.p.x - pos.x, cent.p.y - pos.y);
    Vec2::from_angle(PI / 2.0).rotate(g_v.normalize()) * (G * cent.m / g_v.length()).sqrt()
}

fn rand_initial_pos(cent: PhysicsEntity) -> Vec2 {
    cent.p
        + vec2(
            gen_range(cent.sz, WORLD_SIZE),
            gen_range(cent.sz, WORLD_SIZE),
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
fn calc_orbital_control(p: Vec2, v: Vec2, target_p: Vec2, target_v: Vec2) -> Vec2 {
    // project inputs into polar coords
    let p_pol = PVec2::from_cart(p);
    let target_p_pol = PVec2::from_cart(target_p);
    let v_pol = PVec2::from_cart_vel(p, v);
    let target_v_pol = PVec2::from_cart_vel(target_p, target_v);
    let delta_p = target_p_pol - p_pol;
    let delta_v = target_v_pol - v_pol;

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

    let (free_tx, free_rx) = entity_queue::<Idx>();

    // ENTITIES
    let mut ships = Vec::<Ship>::new();
    let mut asteroids = Vec::<Asteroid>::new();
    let mut planets = Vec::<Planet>::new();
    let mut projectiles = Vec::<Projectile>::new();

    //PHYSICS
    let mut bodies = EntityList::<PhysicsEntity>::new();
    let mut forces = EntityList::<Vec2>::new();
    let (collide_tx, collide_rx) = entity_queue::<(EID, Idx, PhysicsEntity)>();

    // SUN
    let sun = {
        let (phy, planet) = sun();
        planets.put(planet);
        bodies.put(EID::Planet, phy);
        forces.put(EID::Planet, Vec2::ZERO);
        phy
    };

    // PLAYER
    {
        let (phy, ship) = rand_drone(sun);
        ships.put(ship);
        bodies.put(EID::Ship, phy);
        forces.put(EID::Ship, Vec2::ZERO);
    }
    let mut player_idx = 0;

    // SHIPS
    for _i in 0..100 {
        let (phy, ship) = rand_drone(sun);
        ships.put(ship);
        bodies.put(EID::Ship, phy);
        forces.put(EID::Ship, Vec2::ZERO);
    }

    // ASTEROIDS
    for _i in 0..1000 {
        let (phy, asteroid) = rand_asteroid(sun);
        asteroids.put(asteroid);
        bodies.put(EID::Asteroid, phy);
        forces.put(EID::Asteroid, Vec2::ZERO);
    }

    // PLANETS
    for _i in 0..2 {
        let (phy, planet) = rand_planet(sun);
        planets.put(planet);
        bodies.put(EID::Planet, phy);
        forces.put(EID::Planet, Vec2::ZERO);
    }

    let mut input_mag: Vec2;
    let mut input_angle: f32 = 0.0;
    loop {
        // RESET FRAME
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

        // SHIP DECISION
        let m = bodies[EID::Ship][player_idx].m;
        forces[EID::Ship][player_idx] += input_force.normalize_or_zero() * ENGINE / m;

        if is_key_down(Space) {
            let impulse = input_dir * 10.;
            projectiles.push(Projectile {
                state: 0,
                timer: 0,
                lifespan: 0,
            });
            bodies.put(
                EID::Projectile,
                PhysicsEntity {
                    p: bodies[EID::Ship][player_idx].p + impulse * DT * DT,
                    v: bodies[EID::Ship][player_idx].v,
                    sz: 0.05,
                    m: 0.1,
                },
            );
            forces.put(EID::Projectile, impulse);
        }

        // STATE
        ships
            .par_iter_mut()
            .zip(forces[EID::Ship].par_iter_mut())
            .enumerate()
            .for_each(|(idx, (ship, f))| {
                if ship.pressure < 0. {
                    free_tx[EID::Ship].send(idx as u32).unwrap();
                }
            });

        asteroids
            .par_iter_mut()
            .zip(forces[EID::Asteroid].par_iter_mut())
            .enumerate()
            .for_each(|(idx, (asteroid, f))| {
                if asteroid.integrity < 0. {
                    free_tx[EID::Asteroid].send(idx as u32).unwrap();
                }
            });

        // ACTIVE PHYSICS UPDATE
        bodies
            .lists
            .par_iter_mut()
            .zip(forces.lists.par_iter())
            .flatten()
            .for_each(|(entity, f)| {
                let f = *f + gravity_force(entity.p, sun);

                let mut candidate_v = entity.v + f * DT;
                // candidate X first, then Y
                let mut candidate_p = entity.p + vec2(candidate_v.x, 0.) * DT;

                // X check collisions with static geometry
                let mut did_collide: bool = false;
                static_bodies.lists.iter().flatten().for_each(|cl| {
                    let vec = cl.p - candidate_p;
                    let dist = vec.length();
                    let collide_dist = cl.sz + entity.sz;
                    let dir = vec.normalize_or_zero();

                    if dist <= collide_dist {
                        candidate_v.x = cl.v.x;
                        did_collide = true;
                    }
                });

                // Y check collisions with static geometry
                candidate_p += vec2(0., candidate_v.y) * DT;
                static_bodies.lists.iter().flatten().for_each(|cl| {
                    let vec = cl.p - candidate_p;
                    let dist = vec.length();
                    let collide_dist = cl.sz + entity.sz;
                    let dir = vec.normalize_or_zero();

                    if dist < collide_dist {
                        candidate_v.y = cl.v.y;
                        did_collide = true;
                    }
                });

                entity.v = candidate_v;
                entity.p = entity.p + entity.v * DT;
            });

        // CLEAR FORCES
        for li in forces.lists.iter_mut() {
            for f in li.iter_mut() {
                *f = Vec2::ZERO;
            }
        }

        // ACTIVE-ACTIVE COLLISION EVENT HANDLING
        // DETECT
        // NOTE: loop is constructed like this since we need to track the eids and indices, normal
        // .enumerate() doesn't work
        for (eid, li) in active_bodies.lists.iter().enumerate() {
            let eid = EID::from(eid);
            li.par_iter().enumerate().for_each(|(idx, a)| {
                for (collide_eid, collide_li) in active_bodies.lists.iter().enumerate() {
                    let collide_eid = EID::from(collide_eid);
                    collide_li
                        .par_iter()
                        .enumerate()
                        .for_each(|(collide_idx, b)| {
                            let vec = a.p - b.p;
                            let dist = vec.length();
                            let collide_dist = a.sz + b.sz;
                            let is_same = collide_idx == idx && collide_eid as u32 == eid as u32;
                            if dist < collide_dist && !is_same {
                                collide_tx[eid].send((collide_eid, idx as u32, *b)).unwrap();
                            }
                        });
                }
            });
        }

        // HANDLE COLLISIONS OF SHIP
        let my_eid = EID::Ship;
        while let Ok((eid, idx, b)) = collide_rx[my_eid].try_recv() {
            let idx = idx as usize;
            let a = active_bodies[my_eid][idx];
            let relative_velocity = b.v - a.v;
            let dir = (b.p - a.p).normalize_or_zero();
            let collision_force =
                (dir.dot(relative_velocity) / relative_velocity.length()) * b.m * dir / a.m;
            match eid {
                EID::Projectile => {
                    ships[idx].pressure -= 10.;
                    // forces[my_eid][idx] = collision_force;
                }
                _ => {}
            }
        }

        // HANDLE COLLISIONS OF PLANET
        let my_eid = EID::Planet;
        while let Ok((eid, idx, b)) = collide_rx[my_eid].try_recv() {}

        // HANDLE COLLISIONS OF ASTEROIDS
        let my_eid = EID::Asteroid;
        while let Ok((eid, idx, b)) = collide_rx[my_eid].try_recv() {
            let idx = idx as usize;
            let a = active_bodies[my_eid][idx];
            let relative_velocity = b.v - a.v;
            let dir = (b.p - a.p).normalize_or_zero();
            let collision_force =
                (dir.dot(relative_velocity) / relative_velocity.length()) * b.m * dir / a.m;
            match eid {
                EID::Projectile => {
                    asteroids[idx].integrity -= 10.;
                    // forces[my_eid][idx] = collision_force;
                }
                _ => {}
            }
        }

        // HANDLE COLLISIONS OF PROJECTILE
        let my_eid = EID::Projectile;
        while let Ok((eid, idx, b)) = collide_rx[my_eid].try_recv() {
            let idx = idx as usize;
            let a = active_bodies[my_eid][idx];
            let relative_velocity = b.v - a.v;
            let dir = (b.p - a.p).normalize_or_zero();
            let collision_force =
                (dir.dot(relative_velocity) / relative_velocity.length()) * b.m * dir / a.m;

            match eid {
                EID::Ship | EID::Planet | EID::Asteroid => free_tx[my_eid]
                    .send(idx as u32)
                    .expect("failed to add to free_queue"),
                EID::Projectile => {}
            };
        }

        // KILL ENTITIES
        for eid in 0..free_rx.rx.len() {
            let eid = EID::from(eid);
            match eid {
                EID::Ship => {
                    let mut marked: Vec<bool> = vec![true; ships.len()];
                    while let Ok(idx) = free_rx[eid].try_recv() {
                        marked[idx as usize] = false;
                    }

                    let mut iter = marked.iter();
                    ships.retain(|_| *iter.next().unwrap());
                    let mut iter = marked.iter();
                    active_bodies[EID::Ship].retain(|_| *iter.next().unwrap());

                    let mut iter = marked.iter();
                    forces[EID::Ship].retain(|_| *iter.next().unwrap());
                }
                EID::Planet => {
                    let mut marked: Vec<bool> = vec![true; planets.len()];
                    while let Ok(idx) = free_rx[eid].try_recv() {
                        marked[idx as usize] = false;
                    }

                    let mut iter = marked.iter();
                    planets.retain(|_| *iter.next().unwrap());
                    let mut iter = marked.iter();
                    static_bodies[EID::Planet].retain(|_| *iter.next().unwrap());

                    let mut iter = marked.iter();
                    forces[EID::Planet].retain(|_| *iter.next().unwrap());
                }
                EID::Asteroid => {
                    let mut marked: Vec<bool> = vec![true; asteroids.len()];
                    while let Ok(idx) = free_rx[eid].try_recv() {
                        marked[idx as usize] = false;
                    }

                    let mut iter = marked.iter();
                    asteroids.retain(|_| *iter.next().unwrap());
                    let mut iter = marked.iter();
                    active_bodies[EID::Asteroid].retain(|_| *iter.next().unwrap());

                    let mut iter = marked.iter();
                    forces[EID::Asteroid].retain(|_| *iter.next().unwrap());
                }
                EID::Projectile => {
                    let mut marked: Vec<bool> = vec![true; projectiles.len()];
                    while let Ok(idx) = free_rx[eid].try_recv() {
                        marked[idx as usize] = false;
                    }

                    let mut iter = marked.iter();
                    projectiles.retain(|_| *iter.next().unwrap());
                    let mut iter = marked.iter();
                    active_bodies[EID::Projectile].retain(|_| *iter.next().unwrap());

                    let mut iter = marked.iter();
                    forces[EID::Projectile].retain(|_| *iter.next().unwrap());
                }
            }
        }

        // DRAW WORLD
        let camera = {
            let pos = active_bodies[EID::Ship][player_idx].p;
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
        draw_sphere(vec3(sun.p.x, H, sun.p.y), sun.sz, None, YELLOW);

        for (ship, phy) in ships.iter().zip(active_bodies[EID::Ship].iter()) {
            let pos = vec3(phy.p.x, H, phy.p.y);
            draw_line_3d(pos, pos + vec3(input_dir.x, 0., input_dir.y), GREEN);

            draw_cube(pos, Vec3::splat(phy.sz), None, ship.color);

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

        for (planet, phy) in planets.iter().zip(static_bodies[EID::Planet].iter()) {
            draw_sphere(vec3(phy.p.x, H, phy.p.y), phy.sz, None, planet.color);
        }

        for (asteroid, phy) in asteroids.iter().zip(active_bodies[EID::Asteroid].iter()) {
            draw_cube(
                vec3(phy.p.x, H, phy.p.y),
                Vec3::splat(phy.sz),
                None,
                asteroid.color,
            );
        }

        for (projectile, phy) in projectiles
            .iter()
            .zip(active_bodies[EID::Projectile].iter())
        {
            draw_cube(vec3(phy.p.x, H, phy.p.y), Vec3::splat(phy.sz), None, RED);
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

        next_frame().await
    }
}
