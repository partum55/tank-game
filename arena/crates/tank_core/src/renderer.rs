use crate::arena::TankArena;
use crate::math::{to_local, Vec2};
use crate::types::{BulletState, TankState};

/// `color` with every channel scaled by `f` (darker tracks).
fn shade(color: u32, f: f32) -> u32 {
    let ch = |shift: u32| ((((color >> shift) & 0xFF) as f32 * f) as u32).min(0xFF) << shift;
    ch(16) | ch(8) | ch(0)
}

#[cfg(feature = "gui")]
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};

/// Height of the control strip above the field. The strip is NOT part of the arena:
/// the window is `height + HUD_HEIGHT` tall and field coordinates start below the strip.
pub const HUD_HEIGHT: usize = 40;
const BUTTON: usize = 28;

#[derive(Debug, Clone, Copy)]
struct Rect {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
}

impl Rect {
    fn contains(&self, (px, py): (f32, f32)) -> bool {
        px >= self.x as f32
            && px < (self.x + self.w) as f32
            && py >= self.y as f32
            && py < (self.y + self.h) as f32
    }
}

pub struct ArenaRenderer {
    /// Field (arena) size in pixels; the buffer is `width * (height + HUD_HEIGHT)`.
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<u32>,
    y_off: isize,
    paused: bool,
    quit: bool,
    mouse: Option<(f32, f32)>,
    #[cfg(feature = "gui")]
    prev_down: bool,
    #[cfg(feature = "gui")]
    window: Option<Window>,
}

impl ArenaRenderer {
    pub fn new_headless(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buffer: vec![0; width * (height + HUD_HEIGHT)],
            y_off: HUD_HEIGHT as isize,
            paused: false,
            quit: false,
            mouse: None,
            #[cfg(feature = "gui")]
            prev_down: false,
            #[cfg(feature = "gui")]
            window: None,
        }
    }

    #[cfg(feature = "gui")]
    pub fn new_window(width: usize, height: usize, title: &str) -> Result<Self, String> {
        let mut opts = WindowOptions::default();
        opts.resize = false;
        opts.scale = minifb::Scale::X1;

        let mut window = Window::new(title, width, height + HUD_HEIGHT, opts)
            .map_err(|e| format!("Failed to create window: {:?}", e))?;
        window.set_target_fps(60);

        let mut r = Self::new_headless(width, height);
        r.window = Some(window);
        Ok(r)
    }

    #[inline]
    pub fn buffer(&self) -> &[u32] {
        &self.buffer
    }

    fn pause_button(&self) -> Rect {
        Rect { x: self.width.saturating_sub(76), y: (HUD_HEIGHT - BUTTON) / 2, w: BUTTON, h: BUTTON }
    }

    fn quit_button(&self) -> Rect {
        Rect { x: self.width.saturating_sub(40), y: (HUD_HEIGHT - BUTTON) / 2, w: BUTTON, h: BUTTON }
    }

    pub fn is_open(&self) -> bool {
        #[cfg(feature = "gui")]
        if let Some(ref win) = self.window {
            return !self.quit && win.is_open() && !win.is_key_down(Key::Escape);
        }
        !self.quit
    }

    pub fn draw_arena(&mut self, arena: &TankArena) {
        // Field coordinates start below the control strip
        self.y_off = HUD_HEIGHT as isize;

        // 1. Clear background (Dark modern terminal background #111418)
        self.clear(0x00111418);

        // 2. Draw subtle grid
        self.draw_grid(50, 0x001a1f26);

        // 3. Draw arena boundary walls
        self.draw_rect(2, 2, self.width - 4, self.height - 4, 0x00334155, false);

        // 3b. Draw Obstacles (Partition walls)
        for obs in arena.obstacles() {
            let x = (obs.x as usize).min(self.width);
            let y = (obs.y as usize).min(self.height);
            let w = (obs.width as usize).min(self.width.saturating_sub(x));
            let h = (obs.height as usize).min(self.height.saturating_sub(y));
            // Fill obstacle with solid slate
            self.draw_rect(x, y, w, h, 0x002A324B, true);
            // Border with highlight
            self.draw_rect(x, y, w, h, 0x0064748B, false);
        }

        // 4. Draw Bullets
        for bullet in arena.bullets() {
            if bullet.is_active {
                self.draw_bullet(bullet);
            }
        }

        // 5. Draw Tanks
        let (t1, t2) = arena.get_states();
        let (half, max_ammo) = (arena.half_hull(), arena.config.max_ammo);
        self.draw_tank(&t1, half, max_ammo, 0x0000E5FF, 0x0000B4D8); // Cyan for Tank 1
        self.draw_tank(&t2, half, max_ammo, 0x00FF3D00, 0x00D00000); // Orange/Red for Tank 2

        // 6. Control strip (health bars, progress, buttons) - drawn last, covers the strip only
        self.draw_hud(&t1, &t2, arena.step_count, arena.config.max_steps, max_ammo);
    }

    /// Draws one frame and handles input. Returns false once the user quits
    /// (QUIT button, Esc, or the window's own close button). While paused it keeps
    /// redrawing and does not return, so the caller's simulation loop stops with it.
    #[cfg(feature = "gui")]
    pub fn render_window(&mut self, arena: &TankArena) -> bool {
        loop {
            if !self.frame(arena) {
                return false;
            }
            if !self.paused {
                return true;
            }
        }
    }

    #[cfg(feature = "gui")]
    fn frame(&mut self, arena: &TankArena) -> bool {
        let (mouse, down, space, esc, open) = match self.window {
            Some(ref w) => (
                w.get_mouse_pos(MouseMode::Discard),
                w.get_mouse_down(MouseButton::Left),
                w.is_key_pressed(Key::Space, KeyRepeat::No),
                w.is_key_down(Key::Escape),
                w.is_open(),
            ),
            None => return false,
        };
        self.mouse = mouse;
        let click = down && !self.prev_down;
        self.prev_down = down;
        if click {
            if let Some(m) = mouse {
                if self.quit_button().contains(m) {
                    self.quit = true;
                } else if self.pause_button().contains(m) {
                    self.paused = !self.paused;
                }
            }
        }
        if space {
            self.paused = !self.paused;
        }
        if !open || esc || self.quit {
            self.quit = true;
            return false;
        }

        self.draw_arena(arena);
        if let Some(ref mut win) = self.window {
            let _ = win.update_with_buffer(&self.buffer, self.width, self.height + HUD_HEIGHT);
        }
        true
    }

    // --- Drawing primitives ---

    #[inline]
    pub fn clear(&mut self, color: u32) {
        self.buffer.fill(color);
    }

    #[inline]
    pub fn set_pixel(&mut self, x: isize, y: isize, color: u32) {
        let y = y + self.y_off;
        if x >= 0 && x < self.width as isize && y >= 0 && y < (self.height + HUD_HEIGHT) as isize {
            self.buffer[y as usize * self.width + x as usize] = color;
        }
    }

    fn draw_grid(&mut self, step: usize, color: u32) {
        for x in (0..self.width).step_by(step) {
            for y in 0..self.height {
                self.set_pixel(x as isize, y as isize, color);
            }
        }
        for y in (0..self.height).step_by(step) {
            for x in 0..self.width {
                self.set_pixel(x as isize, y as isize, color);
            }
        }
    }

    fn draw_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: u32, fill: bool) {
        if fill {
            for dy in 0..h {
                for dx in 0..w {
                    self.set_pixel((x + dx) as isize, (y + dy) as isize, color);
                }
            }
        } else {
            for dx in 0..w {
                self.set_pixel((x + dx) as isize, y as isize, color);
                self.set_pixel((x + dx) as isize, (y + h.saturating_sub(1)) as isize, color);
            }
            for dy in 0..h {
                self.set_pixel(x as isize, (y + dy) as isize, color);
                self.set_pixel((x + w.saturating_sub(1)) as isize, (y + dy) as isize, color);
            }
        }
    }

    fn draw_circle(&mut self, cx: f32, cy: f32, radius: f32, color: u32, fill: bool) {
        let r = radius as isize;
        let r2 = (radius * radius) as isize;
        let icx = cx as isize;
        let icy = cy as isize;

        for dy in -r..=r {
            for dx in -r..=r {
                let dist_sq = dx * dx + dy * dy;
                if fill {
                    if dist_sq <= r2 {
                        self.set_pixel(icx + dx, icy + dy, color);
                    }
                } else if (dist_sq - r2).abs() <= r * 2 {
                    self.set_pixel(icx + dx, icy + dy, color);
                }
            }
        }
    }

    fn draw_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: u32) {
        let mut x = x0;
        let mut y = y0;
        let dx = x1 - x0;
        let dy = y1 - y0;
        let steps = dx.abs().max(dy.abs()).ceil() as usize;

        if steps == 0 {
            self.set_pixel(x as isize, y as isize, color);
            return;
        }

        let x_inc = dx / steps as f32;
        let y_inc = dy / steps as f32;

        for _ in 0..=steps {
            self.set_pixel(x as isize, y as isize, color);
            x += x_inc;
            y += y_inc;
        }
    }

    /// Draws the hull exactly as the physics sees it: a `2*half.0` x `2*half.1` rectangle turned by `body_angle`.
    fn draw_tank(&mut self, tank: &TankState, half: (f32, f32), max_ammo: u32, color_body: u32, color_turret: u32) {
        let (hl, hw) = half;
        let (s, c) = tank.body_angle.sin_cos();
        let (ex, ey) = (c.abs() * hl + s.abs() * hw, s.abs() * hl + c.abs() * hw);
        let to_world = |lx: f32, ly: f32| Vec2::new(tank.pos.x + lx * c - ly * s, tank.pos.y + lx * s + ly * c);
        let (x0, x1) = ((tank.pos.x - ex).floor() as isize, (tank.pos.x + ex).ceil() as isize);
        let (y0, y1) = ((tank.pos.y - ey).floor() as isize, (tank.pos.y + ey).ceil() as isize);

        if !tank.is_alive {
            // Destroyed wreck: dark hull with a red X
            for py in y0..=y1 {
                for px in x0..=x1 {
                    let l = to_local(Vec2::new(px as f32 + 0.5, py as f32 + 0.5), tank.pos, tank.body_angle);
                    if l.x.abs() <= hl && l.y.abs() <= hw {
                        self.set_pixel(px, py, 0x004A1515);
                    }
                }
            }
            for (a, b) in [((-hl, -hw), (hl, hw)), ((-hl, hw), (hl, -hw))] {
                let (p, q) = (to_world(a.0 * 0.8, a.1 * 0.8), to_world(b.0 * 0.8, b.1 * 0.8));
                self.draw_line(p.x, p.y, q.x, q.y, 0x00FF3333);
            }
            return;
        }

        // 1. Hull: rectangle, darker tracks along the long sides, white outline
        let tracks = shade(color_body, 0.5);
        for py in y0..=y1 {
            for px in x0..=x1 {
                let l = to_local(Vec2::new(px as f32 + 0.5, py as f32 + 0.5), tank.pos, tank.body_angle);
                if l.x.abs() <= hl && l.y.abs() <= hw {
                    let color = if l.x.abs() > hl - 1.0 || l.y.abs() > hw - 1.0 {
                        0x00FFFFFF
                    } else if l.y.abs() > hw - 5.0 {
                        tracks
                    } else {
                        color_body
                    };
                    self.set_pixel(px, py, color);
                }
            }
        }
        // front of the hull: a black bar across the nose
        let (nl, nr) = (to_world(hl - 4.0, -hw + 5.0), to_world(hl - 4.0, hw - 5.0));
        self.draw_line(nl.x, nl.y, nr.x, nr.y, 0x00000000);

        // 2. Turret (round cupola + barrel past the nose, two pixels thick)
        self.draw_circle(tank.pos.x, tank.pos.y, hw * 0.6, color_turret, true);
        self.draw_circle(tank.pos.x, tank.pos.y, hw * 0.6, 0x00FFFFFF, false);
        let dir = Vec2::from_angle(tank.turret_angle);
        let barrel_end = tank.pos + dir * (hl + 4.0);
        for off in [0.0f32, 1.0] {
            let (ox, oy) = (-dir.y * off, dir.x * off);
            self.draw_line(tank.pos.x + ox, tank.pos.y + oy, barrel_end.x + ox, barrel_end.y + oy, 0x00FFFFFF);
        }

        let r = ey;

        // 3. Health bar above tank
        let bar_w = 32;
        let bar_h = 4;
        let bar_x = (tank.pos.x - (bar_w as f32 * 0.5)).max(0.0) as usize;
        let bar_y = (tank.pos.y - r - 10.0).max(4.0) as usize;

        // Background (empty health)
        self.draw_rect(bar_x, bar_y, bar_w, bar_h, 0x00441111, true);

        // Foreground (current HP)
        let hp_ratio = (tank.hp / 100.0).clamp(0.0, 1.0);
        let cur_w = (bar_w as f32 * hp_ratio).round() as usize;
        let hp_color = if hp_ratio > 0.5 {
            0x0000FF66
        } else if hp_ratio > 0.25 {
            0x00FFAA00
        } else {
            0x00FF2222
        };
        self.draw_rect(bar_x, bar_y, cur_w, bar_h, hp_color, true);

        // Ammo bar right below it (thin, amber)
        let ammo_w = (bar_w as f32 * tank.ammo as f32 / max_ammo.max(1) as f32).round() as usize;
        self.draw_rect(bar_x, bar_y + bar_h + 2, bar_w, 3, 0x00332A10, true);
        self.draw_rect(bar_x, bar_y + bar_h + 2, ammo_w, 3, 0x00FFD166, true);
    }

    fn draw_bullet(&mut self, bullet: &BulletState) {
        // Bullet glow
        let color = if bullet.owner_id == 1 {
            0x0000E5FF
        } else {
            0x00FF3D00
        };
        self.draw_circle(bullet.pos.x, bullet.pos.y, bullet.radius, 0x00FFF44F, true);
        // Short trailing line
        let trail = bullet.pos - bullet.vel * 1.5;
        self.draw_line(bullet.pos.x, bullet.pos.y, trail.x, trail.y, color);
    }

    /// Control strip above the field (absolute window coordinates, never overlaps the arena).
    fn draw_hud(&mut self, t1: &TankState, t2: &TankState, step: usize, max_steps: usize, max_ammo: u32) {
        self.y_off = 0;
        self.draw_rect(0, 0, self.width, HUD_HEIGHT, 0x000B0E14, true);
        self.draw_line(0.0, (HUD_HEIGHT - 1) as f32, self.width as f32, (HUD_HEIGHT - 1) as f32, 0x00334155);

        // Tank 1 HP indicator (Left, cyan)
        let t1_bar_w = (150.0 * (t1.hp / 100.0).clamp(0.0, 1.0)) as usize;
        self.draw_rect(20, 14, 150, 12, 0x00331111, true);
        self.draw_rect(20, 14, t1_bar_w, 12, 0x0000E5FF, true);

        // Tank 2 HP indicator (right of centre, before the buttons, orange)
        let t2_bar_w = (150.0 * (t2.hp / 100.0).clamp(0.0, 1.0)) as usize;
        let t2_x = self.width.saturating_sub(76 + 16 + 150);
        self.draw_rect(t2_x, 14, 150, 12, 0x00331111, true);
        self.draw_rect(t2_x + (150 - t2_bar_w), 14, t2_bar_w, 12, 0x00FF3D00, true);

        // Ammo bars under the HP bars (amber): one tick per shot left
        let ammo_px = |t: &TankState| (150.0 * t.ammo as f32 / max_ammo.max(1) as f32).round() as usize;
        self.draw_rect(20, 29, 150, 5, 0x00332A10, true);
        self.draw_rect(20, 29, ammo_px(t1), 5, 0x00FFD166, true);
        self.draw_rect(t2_x, 29, 150, 5, 0x00332A10, true);
        self.draw_rect(t2_x + (150 - ammo_px(t2)), 29, ammo_px(t2), 5, 0x00FFD166, true);

        // Step progress bar (Center)
        let center_x = (self.width / 2).saturating_sub(100);
        let progress = (step as f32 / max_steps as f32).clamp(0.0, 1.0);
        self.draw_rect(center_x, 17, 200, 6, 0x001E293B, true);
        self.draw_rect(center_x, 17, (200.0 * progress) as usize, 6, 0x0094A3B8, true);

        // Buttons: pause / resume and quit (icons, the renderer has no font)
        let hover = |r: Rect, m: Option<(f32, f32)>| m.map_or(false, |p| r.contains(p));
        let pb = self.pause_button();
        let qb = self.quit_button();
        let pause_color = if hover(pb, self.mouse) { 0x00334155 } else { 0x001E293B };
        let quit_color = if hover(qb, self.mouse) { 0x00B91C1C } else { 0x00450A0A };
        self.draw_rect(pb.x, pb.y, pb.w, pb.h, pause_color, true);
        self.draw_rect(pb.x, pb.y, pb.w, pb.h, 0x0064748B, false);
        self.draw_rect(qb.x, qb.y, qb.w, qb.h, quit_color, true);
        self.draw_rect(qb.x, qb.y, qb.w, qb.h, 0x0064748B, false);

        let (pcx, pcy) = ((pb.x + pb.w / 2) as f32, (pb.y + pb.h / 2) as f32);
        if self.paused {
            // play triangle
            for dy in -8i32..=8 {
                let half = 8 - dy.abs();
                self.draw_line(pcx - 5.0, pcy + dy as f32, pcx - 5.0 + half as f32 * 1.4, pcy + dy as f32, 0x0000FF66);
            }
        } else {
            self.draw_rect(pb.x + 8, pb.y + 7, 4, 14, 0x00E2E8F0, true);
            self.draw_rect(pb.x + 16, pb.y + 7, 4, 14, 0x00E2E8F0, true);
        }
        let (qcx, qcy) = ((qb.x + qb.w / 2) as f32, (qb.y + qb.h / 2) as f32);
        for d in [-1.0f32, 0.0, 1.0] {
            self.draw_line(qcx - 7.0 + d, qcy - 7.0, qcx + 7.0 + d, qcy + 7.0, 0x00FFFFFF);
            self.draw_line(qcx - 7.0 + d, qcy + 7.0, qcx + 7.0 + d, qcy - 7.0, 0x00FFFFFF);
        }

        self.y_off = HUD_HEIGHT as isize;
    }
}
