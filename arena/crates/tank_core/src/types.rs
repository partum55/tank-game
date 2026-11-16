use crate::math::Vec2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArenaConfig {
    pub width: f32,
    pub height: f32,
    pub max_steps: usize,
    pub tank_radius: f32,
    pub tank_max_speed: f32,
    pub tank_max_reverse_speed: f32,
    pub tank_accel: f32,
    pub tank_turn_speed: f32,
    pub turret_turn_speed: f32,
    pub bullet_speed: f32,
    pub bullet_damage: f32,
    pub bullet_radius: f32,
    pub reload_ticks: usize,
    /// Shots per tank per match (no resupply).
    pub max_ammo: u32,
    /// Hull rectangle: `tank_length` along the body heading, `tank_width` across.
    /// `tank_radius` is kept only as the map generator's clearance unit (3 radii > hull diagonal).
    pub tank_length: f32,
    pub tank_width: f32,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            width: 1200.0,
            height: 800.0,
            max_steps: 1500,
            tank_radius: 18.0,
            tank_max_speed: 4.5,
            tank_max_reverse_speed: 2.5,
            tank_accel: 0.35,
            tank_turn_speed: 0.05,
            turret_turn_speed: 0.07,
            bullet_speed: 9.0,
            bullet_damage: 25.0,
            bullet_radius: 3.5,
            reload_ticks: 20,
            max_ammo: 20,
            tank_length: 40.0,
            tank_width: 26.0,
        }
    }
}

/// Continuous action output from neural network or bot:
/// All fields are normalized or clamped to [-1.0, 1.0]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TankAction {
    pub thrust: f32,       // [-1.0 .. 1.0] forward/backward
    pub turn: f32,         // [-1.0 .. 1.0] body rotation left/right
    pub turret_turn: f32,  // [-1.0 .. 1.0] turret relative rotation left/right
    pub shoot: bool,       // true if output > 0.5
}

impl TankAction {
    pub fn from_slice(slice: &[f32]) -> Self {
        let thrust = slice.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let turn = slice.get(1).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let turret_turn = slice.get(2).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let shoot_val = slice.get(3).copied().unwrap_or(-1.0);
        let shoot = shoot_val > 0.5;
        Self {
            thrust,
            turn,
            turret_turn,
            shoot,
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.thrust,
            self.turn,
            self.turret_turn,
            if self.shoot { 1.0 } else { -1.0 },
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TankState {
    pub id: u8,
    pub pos: Vec2,
    pub vel: Vec2,
    pub speed: f32,
    pub body_angle: f32,
    pub turret_angle: f32,
    pub hp: f32,
    pub reload_cooldown: usize,
    pub hits_dealt: u32,
    pub hits_taken: u32,
    pub shots_fired: u32,
    /// Bullets that ended on a wall / the border (or a muzzle blocked by a wall) without hitting the enemy.
    pub shots_missed: u32,
    pub ammo: u32,
    pub is_alive: bool,
}

impl TankState {
    pub fn new(id: u8, pos: Vec2, body_angle: f32, ammo: u32) -> Self {
        Self {
            id,
            pos,
            vel: Vec2::ZERO,
            speed: 0.0,
            body_angle,
            turret_angle: body_angle,
            hp: 100.0,
            reload_cooldown: 0,
            hits_dealt: 0,
            hits_taken: 0,
            shots_fired: 0,
            shots_missed: 0,
            ammo,
            is_alive: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BulletState {
    pub id: u32,
    pub owner_id: u8,
    pub pos: Vec2,
    pub vel: Vec2,
    pub radius: f32,
    pub is_active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchOutcome {
    InProgress,
    Tank1Win,
    Tank2Win,
    DrawTie,
    DrawTimeout,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepOutcome {
    pub step_count: usize,
    pub done: bool,
    pub outcome: MatchOutcome,
    pub reward1: f32,
    pub reward2: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obstacle {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Obstacle {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    /// Image under a 180° rotation about the centre of a `w` x `h` arena.
    pub fn mirrored(&self, w: f32, h: f32) -> Self {
        Self::new(w - self.x - self.width, h - self.y - self.height, self.width, self.height)
    }

    /// True if `p` is closer than `margin` (Chebyshev) to this rectangle.
    pub fn near(&self, p: Vec2, margin: f32) -> bool {
        p.x > self.x - margin
            && p.x < self.x + self.width + margin
            && p.y > self.y - margin
            && p.y < self.y + self.height + margin
    }

    /// True if the two rectangles are closer than `margin` to each other.
    pub fn overlaps(&self, o: &Obstacle, margin: f32) -> bool {
        self.x - margin < o.x + o.width
            && self.x + self.width + margin > o.x
            && self.y - margin < o.y + o.height
            && self.y + self.height + margin > o.y
    }
}

/// Rich telemetry exported to Python for customizable feature extraction
#[derive(Debug, Clone, PartialEq)]
pub struct TankTelemetry {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub speed: f32,
    pub body_angle: f32,
    pub turret_angle: f32,
    pub hp: f32,
    pub reload_cooldown: f32,
    /// Shots left / `max_ammo`.
    pub ammo: f32,
    // Enemy-relative info
    pub enemy_rel_x: f32,
    pub enemy_rel_y: f32,
    pub enemy_dist: f32,
    pub enemy_angle_to: f32,
    pub enemy_turret_angle: f32,
    /// Enemy hull heading (live or last known, like `enemy_turret_angle`).
    pub enemy_body_angle: f32,
    pub enemy_hp: f32,
    /// Fog of war: true if the enemy is in direct line of sight. When false, all `enemy_*`
    /// fields above hold the LAST KNOWN values, not the live ones.
    pub enemy_visible: bool,
    /// Ticks since the enemy was last seen, divided by `max_steps` (0.0 while visible).
    pub enemy_last_seen: f32,
    // 8-direction lidar raycast distances (normalized to [0.0 .. 1.0])
    pub lidar_rays: [f32; 8],
    // Static map: (x, y, w, h) of every obstacle, normalized by arena width / height
    pub obstacles: Vec<[f32; 4]>,
    // Bullets telemetry: (rel_x, rel_y, rel_vx, rel_vy, is_enemy)
    pub bullets: Vec<[f32; 5]>,
}
