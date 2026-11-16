pub mod arena;
pub mod mapgen;
pub mod math;
pub mod types;

#[cfg(feature = "gui")]
pub mod renderer;

// Re-exports
pub use arena::TankArena;
pub use math::Vec2;
pub use types::{
    ArenaConfig, BulletState, MatchOutcome, Obstacle, StepOutcome, TankAction, TankState,
    TankTelemetry,
};
