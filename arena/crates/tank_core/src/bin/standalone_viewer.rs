use std::thread::sleep;
use std::time::Duration;
use tank_core::arena::TankArena;
use tank_core::renderer::ArenaRenderer;
use tank_core::types::{ArenaConfig, TankAction};

fn main() {
    println!("=== Tank 2D Battle Arena Viewer (Rust Native) ===");
    println!("Controls: Esc to exit");

    let config = ArenaConfig::default();
    let mut seed: u64 = 42;
    let mut arena = TankArena::new(config, seed);
    let mut renderer = ArenaRenderer::new_window(config.width as usize, config.height as usize, "Tank 2D Battle Arena")
        .expect("Failed to initialize game window");

    let mut tick_counter: usize = 0;

    while renderer.is_open() {
        tick_counter += 1;

        // Simple demo AI logic for Tank 1: Circle and shoot
        let t1_action = TankAction {
            thrust: 0.7,
            turn: 0.3,
            turret_turn: 0.6,
            shoot: tick_counter % 25 == 0,
        };

        // Simple demo AI logic for Tank 2: Aim at Tank 1 and charge
        let (t1, t2) = arena.get_states();
        let to_enemy = t1.pos - t2.pos;
        let enemy_angle = to_enemy.to_angle();
        let angle_diff = (enemy_angle - t2.body_angle).sin();
        let turret_diff = (enemy_angle - t2.turret_angle).sin();

        let t2_action = TankAction {
            thrust: 0.6,
            turn: angle_diff.clamp(-1.0, 1.0),
            turret_turn: turret_diff.clamp(-1.0, 1.0),
            shoot: turret_diff.abs() < 0.2 && tick_counter % 20 == 0,
        };

        let outcome = arena.step(t1_action, t2_action);
        if outcome.done {
            println!("Match concluded! Outcome: {:?}. Resetting...", outcome.outcome);
            seed += 1;
            arena.reset(seed);
            sleep(Duration::from_millis(500));
        }

        if !renderer.render_window(&arena) {
            break;
        }

        sleep(Duration::from_millis(16)); // ~60 FPS
    }

    println!("Viewer closed.");
}
