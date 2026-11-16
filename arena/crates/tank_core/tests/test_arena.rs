use tank_core::arena::TankArena;
use tank_core::types::{ArenaConfig, TankAction};

const NOOP: TankAction = TankAction { thrust: 0.0, turn: 0.0, turret_turn: 0.0, shoot: false };
const SHOOT: TankAction = TankAction { thrust: 0.0, turn: 0.0, turret_turn: 0.0, shoot: true };

/// Arena with a known-empty map, for physics tests independent of the random layout.
fn empty_arena() -> TankArena {
    let mut arena = TankArena::new(ArenaConfig::default(), 42);
    arena.clear_obstacles();
    arena
}

#[test]
fn test_arena_initialization_and_step() {
    let mut arena = TankArena::new(ArenaConfig::default(), 42);
    let (t1, t2) = arena.get_states();
    assert_eq!(t1.hp, 100.0);
    assert_eq!(t2.hp, 100.0);

    let action = TankAction { thrust: 1.0, ..NOOP };
    let outcome = arena.step(action, action);
    assert_eq!(outcome.step_count, 1);
    assert!(!outcome.done);
}

#[test]
fn test_bullet_damage_and_cooldown() {
    let mut config = ArenaConfig::default();
    config.reload_ticks = 10;
    config.bullet_damage = 25.0;
    let mut arena = TankArena::new(config, 42);
    arena.clear_obstacles();
    arena.set_positions_for_testing(100.0, 100.0, 0.0, 200.0, 100.0, std::f32::consts::PI);
    arena.step(SHOOT, NOOP);
    assert_eq!(arena.bullets().len(), 1);

    for _ in 0..30 {
        arena.step(NOOP, NOOP);
    }
    let (_t1, t2) = arena.get_states();
    assert_eq!(t2.hp, 75.0);
}

#[test]
fn test_asynchronous_parallel_action_updates() {
    let mut arena = empty_arena();
    arena.set_tank_action(1, TankAction { thrust: 1.0, ..NOOP });
    let _outcome = arena.tick();

    let (t1, t2) = arena.get_states();
    assert!(t1.speed > 0.0);
    assert_eq!(t2.speed, 0.0);
}

#[test]
fn test_obstacle_blocks_bullets_and_los() {
    let mut arena = empty_arena();
    arena.add_obstacle(380.0, 100.0, 40.0, 400.0);
    arena.set_positions_for_testing(200.0, 300.0, 0.0, 600.0, 300.0, std::f32::consts::PI);

    assert!(!arena.get_telemetry(1).enemy_visible);

    arena.step(SHOOT, NOOP);
    assert_eq!(arena.bullets().len(), 1);
    for _ in 0..50 {
        arena.step(NOOP, NOOP);
    }
    let (_t1, t2) = arena.get_states();
    assert_eq!(t2.hp, 100.0);
    assert_eq!(arena.bullets().len(), 0);
}

#[test]
fn test_same_seed_same_map_different_seed_different_map() {
    let a = TankArena::new(ArenaConfig::default(), 7);
    let b = TankArena::new(ArenaConfig::default(), 7);
    assert_eq!(a.obstacles(), b.obstacles());
    assert_eq!(a.get_states(), b.get_states());

    let differs = (100..120u64).any(|s| {
        let c = TankArena::new(ArenaConfig::default(), s);
        c.obstacles() != a.obstacles() || c.get_states() != a.get_states()
    });
    assert!(differs, "different seeds must produce different episodes");

    // reset(seed) on an existing arena reproduces a fresh one
    let mut r = TankArena::new(ArenaConfig::default(), 1);
    r.reset(7);
    assert_eq!(r.obstacles(), a.obstacles());
    assert_eq!(r.get_states(), a.get_states());
}

#[test]
fn test_maps_are_symmetric_fair_and_passable() {
    let cfg = ArenaConfig::default();
    let clearance = 3.0 * cfg.tank_radius;
    let mut with_obstacles = 0;
    for seed in 0..300u64 {
        let arena = TankArena::new(cfg, seed);
        let (t1, t2) = arena.get_states();
        let obs = arena.obstacles();
        if !obs.is_empty() {
            with_obstacles += 1;
        }

        // Spawns: point mirror of each other, opposite headings, inside the arena.
        assert!((t1.pos.x + t2.pos.x - cfg.width).abs() < 1e-3, "seed {seed}");
        assert!((t1.pos.y + t2.pos.y - cfg.height).abs() < 1e-3, "seed {seed}");
        assert!(t1.pos.x < cfg.width * 0.5);

        // Every obstacle has its mirrored twin.
        for o in obs {
            let m = o.mirrored(cfg.width, cfg.height);
            assert!(
                obs.iter().any(|e| (e.x - m.x).abs() < 1e-3
                    && (e.y - m.y).abs() < 1e-3
                    && (e.width - m.width).abs() < 1e-3
                    && (e.height - m.height).abs() < 1e-3),
                "seed {seed}: obstacle {o:?} has no twin"
            );
            // Gap to border and to spawns (so tanks fit through every passage).
            assert!(o.x >= clearance && o.y >= clearance, "seed {seed}");
            assert!(o.x + o.width <= cfg.width - clearance, "seed {seed}");
            assert!(o.y + o.height <= cfg.height - clearance, "seed {seed}");
            assert!(!o.near(t1.pos, clearance) && !o.near(t2.pos, clearance), "seed {seed}");
        }
        // Gap between distinct obstacles.
        for (i, a) in obs.iter().enumerate() {
            for b in obs.iter().skip(i + 1) {
                assert!(!a.overlaps(b, clearance - 1e-3), "seed {seed}: {a:?} vs {b:?}");
            }
        }
    }
    assert!(with_obstacles > 250, "maps should almost always contain obstacles");
}

#[test]
fn test_fog_of_war_reports_last_known_enemy() {
    let mut arena = empty_arena();
    arena.set_positions_for_testing(200.0, 300.0, 0.0, 600.0, 300.0, std::f32::consts::PI);
    let seen = arena.get_telemetry(1);
    assert!(seen.enemy_visible);
    assert_eq!(seen.enemy_last_seen, 0.0);

    // Wall appears; tank 2 drives away (up) while hidden.
    arena.add_obstacle(380.0, 100.0, 40.0, 400.0);
    for _ in 0..20 {
        arena.step(NOOP, TankAction { thrust: 1.0, turn: 1.0, ..NOOP });
    }
    let (t1, t2) = arena.get_states();
    assert!((t2.pos.x - 600.0).abs() > 1.0 || (t2.pos.y - 300.0).abs() > 1.0, "enemy moved");

    let hidden = arena.get_telemetry(1);
    assert!(!hidden.enemy_visible);
    assert!(hidden.enemy_last_seen > 0.0);
    // Still reports the position from when it was last seen (600, 300), not the live one.
    let expect_rel_x = (600.0 - t1.pos.x) / arena.config.width;
    let expect_rel_y = (300.0 - t1.pos.y) / arena.config.height;
    assert!((hidden.enemy_rel_x - expect_rel_x).abs() < 1e-4);
    assert!((hidden.enemy_rel_y - expect_rel_y).abs() < 1e-4);
    assert_eq!(hidden.enemy_hp, 100.0);

    // The map is public in the telemetry.
    assert!(hidden.obstacles.iter().any(|o| (o[0] - 380.0 / arena.config.width).abs() < 1e-4));
}

#[test]
fn test_shot_against_wall_does_not_pass_through() {
    let mut arena = empty_arena();
    // Thin wall directly in front of tank 1's muzzle; tank 1 pressed against it.
    arena.add_obstacle(120.0, 250.0, 2.0, 100.0);
    arena.set_positions_for_testing(100.0, 300.0, 0.0, 300.0, 300.0, std::f32::consts::PI);
    arena.step(SHOOT, NOOP);
    assert_eq!(arena.bullets().len(), 0, "muzzle blocked: no bullet");
    for _ in 0..60 {
        arena.step(NOOP, NOOP);
    }
    assert_eq!(arena.get_states().1.hp, 100.0);
}

#[test]
fn test_reset_clears_episode_state() {
    let mut arena = TankArena::new(ArenaConfig::default(), 3);
    arena.step(SHOOT, SHOOT);
    arena.reset(3);
    assert_eq!(arena.step_count, 0);
    assert!(arena.bullets().is_empty());
    let t = arena.get_telemetry(1);
    assert_eq!(t.enemy_last_seen, 0.0);
    assert_eq!(t.ammo, 1.0);
    assert_eq!(arena.get_states().0.ammo, arena.config.max_ammo);
}

#[test]
fn test_ammo_runs_out_after_max_ammo_shots() {
    let mut config = ArenaConfig::default();
    config.max_steps = 5000;
    let mut arena = TankArena::new(config, 42);
    arena.clear_obstacles();
    // Far apart, tank 1 aims along the long diagonal-free row: shots just fly into the wall.
    arena.set_positions_for_testing(100.0, 100.0, std::f32::consts::FRAC_PI_2, 1100.0, 700.0, 0.0);
    for _ in 0..(arena.config.max_ammo as usize + 5) * (arena.config.reload_ticks + 1) {
        arena.step(SHOOT, NOOP);
    }
    let (t1, t2) = arena.get_states();
    assert_eq!(t1.shots_fired, arena.config.max_ammo);
    assert_eq!(t1.ammo, 0);
    assert_eq!(t2.shots_fired, 0);
    assert_eq!(t2.ammo, arena.config.max_ammo);
    assert_eq!(arena.get_telemetry(1).ammo, 0.0);
}

#[test]
fn test_blocked_muzzle_burns_ammo_and_counts_as_miss() {
    let mut arena = empty_arena();
    arena.add_obstacle(120.0, 250.0, 2.0, 100.0);
    arena.set_positions_for_testing(100.0, 300.0, 0.0, 300.0, 300.0, std::f32::consts::PI);
    arena.step(SHOOT, NOOP);
    let t1 = arena.get_states().0;
    assert_eq!(t1.ammo, arena.config.max_ammo - 1);
    assert_eq!(t1.shots_fired, 1);
    assert_eq!(t1.shots_missed, 1);
}

#[test]
fn test_bullet_into_wall_counts_as_miss_and_hit_does_not() {
    let mut arena = empty_arena();
    arena.add_obstacle(400.0, 100.0, 30.0, 400.0);
    arena.set_positions_for_testing(100.0, 300.0, 0.0, 700.0, 300.0, std::f32::consts::PI);
    arena.step(SHOOT, NOOP);
    for _ in 0..80 {
        arena.step(NOOP, NOOP);
    }
    let t1 = arena.get_states().0;
    assert_eq!((t1.shots_fired, t1.shots_missed, t1.hits_dealt), (1, 1, 0));

    arena.clear_obstacles();
    for _ in 0..30 {
        arena.step(NOOP, NOOP);
    }
    arena.step(SHOOT, NOOP);
    for _ in 0..80 {
        arena.step(NOOP, NOOP);
    }
    let t1 = arena.get_states().0;
    assert_eq!((t1.shots_fired, t1.shots_missed, t1.hits_dealt), (2, 1, 1));
}

#[test]
fn test_hull_is_a_rectangle_for_bullets() {
    // Tank 2 stands at y=300 heading along x (long axis horizontal): a bullet flying horizontally at
    // y offset 15 px (> half width 13, < old radius 18) must miss it; the old circle would have hit.
    let mut arena = empty_arena();
    arena.set_positions_for_testing(100.0, 315.0, 0.0, 400.0, 300.0, 0.0);
    arena.step(SHOOT, NOOP);
    for _ in 0..60 {
        arena.step(NOOP, NOOP);
    }
    assert_eq!(arena.get_states().1.hp, 100.0, "bullet passes beside the narrow side");

    // Turned across the bullet's path, the same tank is 40 px tall: a bullet 15 px off the centre hits.
    let mut arena = empty_arena();
    arena.set_positions_for_testing(100.0, 315.0, 0.0, 400.0, 300.0, std::f32::consts::FRAC_PI_2);
    arena.step(SHOOT, NOOP);
    for _ in 0..60 {
        arena.step(NOOP, NOOP);
    }
    assert_eq!(arena.get_states().1.hp, 75.0, "bullet hits the long side");
}

#[test]
fn test_turning_tank_is_pushed_out_of_wall_and_stays_inside_arena() {
    let mut arena = empty_arena();
    // Wall right next to tank 1; keep turning and driving: it must never overlap the wall or leave the arena.
    arena.add_obstacle(140.0, 200.0, 30.0, 200.0);
    arena.set_positions_for_testing(100.0, 300.0, 0.0, 1100.0, 700.0, std::f32::consts::PI);
    let half = arena.half_hull();
    for i in 0..400 {
        let a = TankAction { thrust: 1.0, turn: if i % 40 < 20 { 1.0 } else { -1.0 }, ..NOOP };
        arena.step(a, a);
        for t in [arena.get_states().0, arena.get_states().1] {
            let (ex, ey) = tank_core::math::obb_half_extents(t.body_angle, half.0, half.1);
            assert!(t.pos.x >= ex - 1e-3 && t.pos.x <= arena.config.width - ex + 1e-3);
            assert!(t.pos.y >= ey - 1e-3 && t.pos.y <= arena.config.height - ey + 1e-3);
        }
        let t1 = arena.get_states().0;
        let o = arena.obstacles()[0];
        let push = tank_core::math::obb_rect_mtv(t1.pos, t1.body_angle, half, o.x, o.y, o.width, o.height);
        assert!(push.map_or(true, |v| v.length() < 0.01), "tank 1 overlaps the wall at tick {i}");
    }
}

#[test]
fn test_telemetry_shows_enemy_body_angle_and_lidar_sees_the_hull() {
    let mut arena = empty_arena();
    arena.set_positions_for_testing(100.0, 300.0, 0.0, 500.0, 300.0, std::f32::consts::FRAC_PI_2);
    let t = arena.get_telemetry(1);
    assert!((t.enemy_body_angle - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    // Ray 3 points along the body heading (towards the enemy): it stops at the hull's near side.
    // The enemy heads +y, so across the ray it is 26 wide: near side 13 px before its centre (a circle would say 18).
    let diag = (arena.config.width.powi(2) + arena.config.height.powi(2)).sqrt();
    let expected = (400.0 - 13.0) / diag;
    assert!((t.lidar_rays[3] - expected).abs() < 1e-4, "{} vs {}", t.lidar_rays[3], expected);
}
