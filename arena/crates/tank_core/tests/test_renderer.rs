use tank_core::arena::TankArena;
use tank_core::renderer::{ArenaRenderer, HUD_HEIGHT};
use tank_core::types::ArenaConfig;

#[test]
fn test_renderer_buffer_generation() {
    let arena = TankArena::new(ArenaConfig::default(), 42);
    let cfg = arena.config;
    let (w, h) = (cfg.width as usize, cfg.height as usize);
    let mut renderer = ArenaRenderer::new_headless(w, h);
    renderer.draw_arena(&arena);
    let buf = renderer.buffer();
    // the control strip sits above the field, it is not part of the arena
    assert_eq!(buf.len(), w * (h + HUD_HEIGHT));
    // the HUD strip is fully painted (nothing of the field leaks into it)
    assert!(buf[..w * HUD_HEIGHT].iter().all(|&p| p != 0));
    // field row 0 starts right below the strip and shows the arena, not the HUD colour
    assert_ne!(buf[w * HUD_HEIGHT + w / 2], 0x000B0E14);
    // Ensure screen is not completely blank (black)
    assert!(buf.iter().any(|&p| p != 0x000000));
}

#[test]
fn test_tank_is_drawn_as_the_physics_rectangle() {
    let mut arena = TankArena::new(ArenaConfig::default(), 42);
    arena.clear_obstacles();
    // Tank 1 heading along x at (300, 300): hull is 40 long (x) and 26 wide (y).
    arena.set_positions_for_testing(300.0, 300.0, 0.0, 900.0, 500.0, std::f32::consts::PI);
    let (w, h) = (arena.config.width as usize, arena.config.height as usize);
    let mut renderer = ArenaRenderer::new_headless(w, h);
    renderer.draw_arena(&arena);
    let buf = renderer.buffer();
    let at = |x: usize, y: usize| buf[(y + HUD_HEIGHT) * w + x];
    let background = at(10, 10);
    // nose and tail corners are filled (a circle of radius 18 would not reach x = 300 +- 19 near the corner)
    assert_ne!(at(318, 310), background, "front-right corner area belongs to the hull");
    assert_ne!(at(282, 290), background, "rear-left corner area belongs to the hull");
    // beside the narrow side the old circle (r = 18) would still paint; the rectangle (half width 13) does not
    // (x = 305: not on a grid line; 16.8 px from the centre = inside the old circle, outside the rectangle)
    assert_eq!(at(305, 300 + 16), background, "outside the hull's side");
}
