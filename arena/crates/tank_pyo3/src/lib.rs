use pyo3::prelude::*;
use pyo3::types::PyDict;
use tank_core::arena::TankArena;
#[cfg(feature = "gui")]
use tank_core::renderer::ArenaRenderer;
use tank_core::types::{ArenaConfig, MatchOutcome, StepOutcome, TankAction, TankTelemetry};

#[pyclass(unsendable)]
struct PyTankArena {
    arena: TankArena,
    #[cfg(feature = "gui")]
    renderer: Option<ArenaRenderer>,
    seed: u64,
}

#[pymethods]
impl PyTankArena {
    #[new]
    #[pyo3(signature = (width=1200.0, height=800.0, max_steps=1500, seed=42))]
    fn new(width: f32, height: f32, max_steps: usize, seed: u64) -> Self {
        let mut config = ArenaConfig::default();
        config.width = width;
        config.height = height;
        config.max_steps = max_steps;

        let arena = TankArena::new(config, seed);
        Self {
            arena,
            #[cfg(feature = "gui")]
            renderer: None,
            seed,
        }
    }

    #[pyo3(signature = (seed=None))]
    fn reset<'py>(&mut self, py: Python<'py>, seed: Option<u64>) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        // Explicit seed => reproducible episode; no seed => next map in sequence.
        self.seed = match seed {
            Some(s) => s,
            None => self.seed.wrapping_add(1),
        };
        self.arena.reset(self.seed);

        let t1 = self.arena.get_telemetry(1);
        let t2 = self.arena.get_telemetry(2);

        let d1 = telemetry_to_dict(py, &t1)?;
        let d2 = telemetry_to_dict(py, &t2)?;
        Ok((d1, d2))
    }

    fn set_action(&mut self, tank_id: u8, action: Vec<f32>) {
        let act = TankAction::from_slice(&action);
        self.arena.set_tank_action(tank_id, act);
    }

    fn tick<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>, Vec<f32>, bool, Bound<'py, PyDict>)> {
        let outcome = self.arena.tick();
        self.pack_step_result(py, outcome)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action_t1: Vec<f32>,
        action_t2: Vec<f32>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>, Vec<f32>, bool, Bound<'py, PyDict>)> {
        let a1 = TankAction::from_slice(&action_t1);
        let a2 = TankAction::from_slice(&action_t2);
        let outcome = self.arena.step(a1, a2);
        self.pack_step_result(py, outcome)
    }

    fn add_obstacle(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.arena.add_obstacle(x, y, width, height);
    }

    fn clear_obstacles(&mut self) {
        self.arena.clear_obstacles();
    }

    #[cfg(feature = "gui")]
    fn render(&mut self) -> PyResult<bool> {
        let width = self.arena.config.width as usize;
        let height = self.arena.config.height as usize;

        if self.renderer.is_none() {
            let rend = ArenaRenderer::new_window(width, height, "Tank 2D Battle Arena")
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            self.renderer = Some(rend);
        }

        if let Some(ref mut rend) = self.renderer {
            let is_running = rend.render_window(&self.arena);
            Ok(is_running)
        } else {
            Ok(false)
        }
    }

    #[cfg(not(feature = "gui"))]
    fn render(&mut self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "tank_arena was built without the `gui` feature (headless build): rendering is unavailable",
        ))
    }

    fn close(&mut self) {
        #[cfg(feature = "gui")]
        {
            self.renderer = None;
        }
    }
}

impl PyTankArena {
    fn pack_step_result<'py>(
        &self,
        py: Python<'py>,
        outcome: StepOutcome,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>, Vec<f32>, bool, Bound<'py, PyDict>)> {
        let t1 = self.arena.get_telemetry(1);
        let t2 = self.arena.get_telemetry(2);

        let d1 = telemetry_to_dict(py, &t1)?;
        let d2 = telemetry_to_dict(py, &t2)?;
        let rewards = vec![outcome.reward1, outcome.reward2];

        let info = PyDict::new(py);
        info.set_item("step", outcome.step_count)?;
        let outcome_str = match outcome.outcome {
            MatchOutcome::InProgress => "in_progress",
            MatchOutcome::Tank1Win => "tank1_win",
            MatchOutcome::Tank2Win => "tank2_win",
            MatchOutcome::DrawTie => "draw_tie",
            MatchOutcome::DrawTimeout => "draw_timeout",
        };
        info.set_item("outcome", outcome_str)?;
        // Cumulative per-tank counters [tank1, tank2] (diff two ticks for per-tick events).
        let (a, b) = self.arena.get_states();
        info.set_item("shots_fired", vec![a.shots_fired, b.shots_fired])?;
        info.set_item("shots_missed", vec![a.shots_missed, b.shots_missed])?;
        info.set_item("hits_dealt", vec![a.hits_dealt, b.hits_dealt])?;
        info.set_item("ammo", vec![a.ammo, b.ammo])?;

        Ok((d1, d2, rewards, outcome.done, info))
    }
}

fn telemetry_to_dict<'py>(py: Python<'py>, t: &TankTelemetry) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("x", t.x)?;
    dict.set_item("y", t.y)?;
    dict.set_item("vx", t.vx)?;
    dict.set_item("vy", t.vy)?;
    dict.set_item("speed", t.speed)?;
    dict.set_item("body_angle", t.body_angle)?;
    dict.set_item("turret_angle", t.turret_angle)?;
    dict.set_item("hp", t.hp)?;
    dict.set_item("reload_cooldown", t.reload_cooldown)?;
    dict.set_item("ammo", t.ammo)?;

    dict.set_item("enemy_rel_x", t.enemy_rel_x)?;
    dict.set_item("enemy_rel_y", t.enemy_rel_y)?;
    dict.set_item("enemy_dist", t.enemy_dist)?;
    dict.set_item("enemy_angle_to", t.enemy_angle_to)?;
    dict.set_item("enemy_turret_angle", t.enemy_turret_angle)?;
    dict.set_item("enemy_body_angle", t.enemy_body_angle)?;
    dict.set_item("enemy_hp", t.enemy_hp)?;
    dict.set_item("enemy_visible", t.enemy_visible)?;
    dict.set_item("enemy_last_seen", t.enemy_last_seen)?;

    dict.set_item("lidar_rays", t.lidar_rays.to_vec())?;
    dict.set_item("obstacles", t.obstacles.clone())?;
    dict.set_item("bullets", t.bullets.clone())?;

    Ok(dict)
}

#[pymodule]
fn tank_arena(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTankArena>()?;
    Ok(())
}
