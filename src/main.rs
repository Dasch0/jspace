use macroquad::prelude::*;
use macroquad::ui::{
    hash, root_ui,
    widgets::{self, Group},
    Drag, Ui,
};
use nanoserde::{DeRon, SerRon};
use strum::VariantNames;
use strum_macros::{EnumCount, EnumIter, EnumString, EnumVariantNames, IntoStaticStr};

const DT: f32 = 0.1;
const GRID_SCALE: f32 = 10.;

#[derive(SerRon, DeRon, Default)]
struct Ship {
    p: Vec2,
    v: Vec2,
    f: Vec2,
}

#[derive(
    Default, SerRon, DeRon, EnumString, EnumCount, EnumIter, EnumVariantNames, IntoStaticStr, Clone,
)]
enum ShipTile {
    #[default]
    Ground,
}

#[derive(
    Default, Clone, SerRon, DeRon, EnumString, EnumCount, EnumIter, EnumVariantNames, IntoStaticStr,
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

/// Draw a grid centered at (0, 0, 0)
pub fn draw_grid_2d(slices: u32, spacing: f32, axes_color: Color, other_color: Color) {
    let half_slices = (slices as i32) / 2;
    for i in -half_slices..half_slices + 1 {
        let color = if i == 0 { axes_color } else { other_color };

        draw_line(
            i as f32 * spacing,
            -half_slices as f32 * spacing,
            i as f32 * spacing,
            half_slices as f32 * spacing,
            1.0,
            color,
        );
        draw_line(
            -half_slices as f32 * spacing,
            i as f32 * spacing,
            half_slices as f32 * spacing,
            i as f32 * spacing,
            1.0,
            color,
        );
    }
}

#[macroquad::main("BasicShapes")]
async fn main() {
    // WORLD ELEMENTS
    let grid = vec![vec![ShipTile::Ground; 32]; 32];
    let ship = Ship::default();
    let map = Vec::<Terrain>::new();

    // INPUT ELEMENTS
    let mut scroll_sens = 0.01;
    let mut zoom = 1.0;

    // UI ELEMENTS
    let mut apply = false;
    let mut cancel = false;
    let mut combobox = 0;
    let mut text = String::new();
    let mut number = 0.0;

    loop {
        // USER INPUT
        match mouse_wheel() {
            (_x, y) if y != 0.0 => {
                zoom *= 1.1f32.powf(y);
            }
            _ => (),
        }

        println!("{}", zoom);

        // DRAW WORLD
        clear_background(SKYBLUE);
        set_camera(&Camera2D {
            target: ship.p,
            zoom: vec2(zoom, zoom * screen_width() / screen_height()),
            ..Default::default()
        });

        draw_grid_2d(64, GRID_SCALE, BLACK, GRAY);
        draw_circle(ship.p.x, ship.p.y, 1.0, WHITE);

        for (y, row) in grid.iter().enumerate() {
            for (x, tile) in row.iter().enumerate() {
                let x = x as f32 * GRID_SCALE;
                let y = y as f32 * GRID_SCALE;

                match tile {
                    ShipTile::Ground => {} //draw_rectangle(x, y, GRID_SCALE / 2., GRID_SCALE / 2., RED),
                }
            }
        }

        // DRAW UI
        set_default_camera();
        root_ui().window(hash!(), vec2(250., 20.), vec2(500., 250.), |ui| {
            ui.combo_box(hash!(), "Tile", ShipTile::VARIANTS, &mut combobox);

            apply = widgets::Button::new("Apply")
                .position(vec2(80.0, 150.0))
                .ui(ui);
            cancel = widgets::Button::new("Cancel")
                .position(vec2(280.0, 150.0))
                .ui(ui);
        });

        next_frame().await
    }
}
