use macroquad::{prelude::*, rand::gen_range, prelude::KeyCode::*};
use core::f32::consts::PI;

const DT: f32 = 0.1;
const G: f32 = 1.0;
const M: f32 = 10.0;
const ENGINE: f32 = 0.1;

const WORLD_SIZE: f32 = 50.0;

#[macroquad::main("jspace")]
async fn main() {
    let g_cent = vec2(0.0, 0.0);

    let mut pos: Vec2 = vec2(10.0, 10.0);
    let mut vel = {
        let i_p = pos; 
        let g_v = vec2(g_cent.x-i_p.x, g_cent.y-i_p.y);
        Vec2::from_angle(PI / 2.0).rotate(g_v.normalize()) * (G * M / g_v.length()).sqrt()
    };
    let mut capacity: f32 = 0.0;
    let mut fuel: f32 = 500.0;

    let mut objs: Vec<Vec4> = Vec::new();
    let mut planets: Vec<Vec4> = Vec::new();
    for i in 0..1000 {
        let i_p =  g_cent + vec2(gen_range(1.0, WORLD_SIZE), gen_range(1.0, WORLD_SIZE));
        let g_v = vec2(g_cent.x-i_p.x, g_cent.y-i_p.y);
        let i_v = Vec2::from_angle(PI / 2.0).rotate(g_v.normalize()) * (G * M / g_v.length()).sqrt();
        objs.push(vec4(i_p.x, i_p.y, i_v.x, i_v.y));
    }

    for i in 0..4 {
        let i_p =  g_cent + vec2(gen_range(1.0, WORLD_SIZE), gen_range(1.0, WORLD_SIZE));
        let g_v = vec2(g_cent.x-i_p.x, g_cent.y-i_p.y);
        let i_v = Vec2::from_angle(PI / 2.0).rotate(g_v.normalize()) * (G * M / g_v.length()).sqrt();
        planets.push(vec4(i_p.x, i_p.y, i_v.x, i_v.y));
    }

    let mut input_mag: Vec2;
    let mut input_angle: f32 = 0.0;
    loop {
        // INPUT
        input_mag = Vec2::ZERO;
        if is_key_down(W) && fuel > 0.0 {
            fuel -= 1.0;
            input_mag += Vec2::X
        }
        if is_key_down(A) {input_angle -= PI/128.0}
        if is_key_down(D) {input_angle += PI/128.0}

        let steering_dir = Vec2::from_angle(input_angle).rotate(Vec2::X * 2.0);
        let input = Vec2::from_angle(input_angle).rotate(input_mag);

        // UPDATE
        for planet in planets.iter_mut() {
            let g_v = vec2(g_cent.x-planet.x, g_cent.y-planet.y);
            let a_g = G * M / g_v.length_squared();
            let f_g = a_g * g_v.normalize();
            planet.z += f_g.x * DT;
            planet.w += f_g.y * DT;
            planet.x += planet.z * DT;
            planet.y += planet.w * DT;
        }
        for obj in objs.iter_mut() {
            let g_v = vec2(g_cent.x-obj.x, g_cent.y-obj.y);
            let a_g = G * M / g_v.length_squared();
            let f_g = a_g * g_v.normalize();
            obj.z += f_g.x * DT;
            obj.w += f_g.y * DT;
            obj.x += obj.z * DT;
            obj.y += obj.w * DT;
        }

        let g_v = vec2(g_cent.x-pos.x, g_cent.y-pos.y);
        let a_g = G * M / g_v.length_squared();
        let f_g = a_g * g_v.normalize();
        vel += (f_g + input.normalize_or_zero() * ENGINE) * DT;

        // COLLISION_DET
        // currently don't allow for sliding collisions
        let mut candidate_pos = pos + vel * DT;
        for planet in planets.iter() {
            let dist = candidate_pos.distance(planet.xy());
            if dist <= 0.5 {
                fuel = 500.0;
                vel = planet.zw();
            }
        }
        pos += vel * DT;

        // PICKUP
        for obj in objs.iter_mut() {
            let dist = candidate_pos.distance(obj.xy());
            if dist <= 0.1 {
                capacity += 1.0;
                let i_p =  g_cent + vec2(gen_range(1.0, WORLD_SIZE), gen_range(1.0, WORLD_SIZE));
                let g_v = vec2(g_cent.x-i_p.x, g_cent.y-i_p.y);
                let i_v = Vec2::from_angle(PI / 2.0).rotate(g_v.normalize()) * (G * M / g_v.length()).sqrt();
                *obj = vec4(i_p.x, i_p.y, i_v.x, i_v.y);
            }
        }

        // DRAW
        set_camera(&Camera3D {
            position: vec3(pos.x - steering_dir.x * 5.0, 5., pos.y - steering_dir.y * 5.0),
            up: vec3(0., 1., 0.),
            target: vec3(pos.x + steering_dir.x, 1.0, pos.y + steering_dir.y),
            ..Default::default()
        });
        clear_background(DARKGRAY);
        draw_grid(200, 1., BLACK, GRAY);

        draw_line_3d(vec3(pos.x, 1.0, pos.y), vec3(pos.x + steering_dir.x, 1.0, pos.y + steering_dir.y), GREEN);
        draw_cube(vec3(pos.x, 1.0, pos.y), vec3(0.2, 0.2, 0.2), None, ORANGE);

        draw_cube(vec3(0., 1.0, 0.), vec3(1.0, 1.0, 1.0), None, YELLOW);

        for planet in planets.iter() {
            draw_cube(vec3(planet.x, 1.0, planet.y), vec3(0.5, 0.5, 0.5), None, BLUE);
        }

        for obj in objs.iter() {
            draw_cube(vec3(obj.x, 1.0, obj.y), vec3(0.1, 0.1, 0.1), None, WHITE);
        }

        set_default_camera();
        draw_text(format!("FPS: {}", get_fps()).as_str(), 20.0, 20.0, 20.0, GREEN);

        draw_rectangle(screen_width() - 100., screen_height() - 50. - capacity, 5., capacity, WHITE);
        draw_rectangle(screen_width() - 80., screen_height() - 50. - fuel, 5., fuel, ORANGE);

        next_frame().await
    }
}
