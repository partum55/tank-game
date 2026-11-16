//! Seeded, point-symmetric map + spawn generation.
//!
//! Fairness: the whole layout is symmetric under a 180° rotation about the arena centre.
//! Tank 2 spawns at the mirror image of tank 1 and every obstacle has a mirrored twin,
//! so neither side has a positional advantage.
//!
//! Connectivity: every obstacle keeps a gap of `CLEARANCE_RADII * tank_radius` to the arena
//! border, to every other obstacle and to both spawn points. Obstacles are therefore isolated
//! islands, free space is always connected and a tank always fits through every passage.

use crate::math::Vec2;
use crate::types::{ArenaConfig, Obstacle};

/// Minimal gap (in tank radii) kept around every obstacle.
pub const CLEARANCE_RADII: f32 = 3.0;

/// SplitMix64: tiny, fast, good enough for map generation, no dependency.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

#[derive(Debug, Clone)]
pub struct Layout {
    pub obstacles: Vec<Obstacle>,
    /// Tank 1 spawn (tank 2 is the point mirror of it).
    pub pos1: Vec2,
    pub angle1: f32,
}

pub fn generate(config: &ArenaConfig, seed: u64) -> Layout {
    let (w, h) = (config.width, config.height);
    let clearance = CLEARANCE_RADII * config.tank_radius;
    let mut rng = SplitMix64::new(seed);

    // Spawn: left zone, facing roughly towards the centre.
    let pos1 = Vec2::new(rng.range(0.08 * w, 0.22 * w), rng.range(0.15 * h, 0.85 * h));
    let angle1 = rng.range(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
    let spawns = [pos1, Vec2::new(w - pos1.x, h - pos1.y)];

    let mut obstacles: Vec<Obstacle> = Vec::new();

    // Optional self-symmetric block in the middle.
    if rng.next_f32() < 0.5 {
        let (bw, bh) = (rng.range(50.0, 120.0), rng.range(50.0, 120.0));
        let block = Obstacle::new(0.5 * w - 0.5 * bw, 0.5 * h - 0.5 * bh, bw, bh);
        if fits(&block, None, &obstacles, &spawns, clearance, w, h) {
            obstacles.push(block);
        }
    }

    // 2..=5 mirrored pairs of walls.
    let pairs = 2 + (rng.next_u64() % 4) as usize;
    for _ in 0..pairs {
        for _attempt in 0..100 {
            let length = rng.range(90.0, 260.0);
            let thickness = rng.range(20.0, 40.0);
            let horizontal = rng.next_f32() < 0.5;
            let (ow, oh) = if horizontal { (length, thickness) } else { (thickness, length) };
            let x = rng.range(0.25 * w, (0.75 * w - ow).max(0.25 * w));
            let y = rng.range(clearance, (h - clearance - oh).max(clearance));
            let wall = Obstacle::new(x, y, ow, oh);
            let twin = wall.mirrored(w, h);
            if fits(&wall, Some(&twin), &obstacles, &spawns, clearance, w, h) {
                obstacles.push(wall);
                obstacles.push(twin);
                break;
            }
        }
    }

    Layout { obstacles, pos1, angle1 }
}

fn fits(
    o: &Obstacle,
    twin: Option<&Obstacle>,
    existing: &[Obstacle],
    spawns: &[Vec2; 2],
    clearance: f32,
    w: f32,
    h: f32,
) -> bool {
    if o.x - clearance < 0.0
        || o.y - clearance < 0.0
        || o.x + o.width + clearance > w
        || o.y + o.height + clearance > h
    {
        return false;
    }
    if existing.iter().any(|e| o.overlaps(e, clearance)) {
        return false;
    }
    if let Some(t) = twin {
        if o.overlaps(t, clearance) || existing.iter().any(|e| t.overlaps(e, clearance)) {
            return false;
        }
    }
    !spawns.iter().any(|s| o.near(*s, clearance))
}
