"""Evaluation on the real Rust engine, spread over CPU processes (the GPU trainer leaves the vCPUs idle anyway)."""
import multiprocessing as mp


def _play(job):
    kind, params_a, params_b, seeds = job
    import referee
    from tank_env import RuleBasedBot
    from mytrain.gru_policy import NumpyGRUPolicy

    a = NumpyGRUPolicy(params_a)
    b = RuleBasedBot() if kind == "bot" else NumpyGRUPolicy(params_b)
    res = referee.play_series(a, b, seeds)
    s = res["A"]
    return s["wins"], s["draws"], res["games"]


class EvalPool:
    def __init__(self, processes: int):
        self.n = max(1, processes)
        self.pool = mp.get_context("spawn").Pool(self.n)

    def scores(self, params, opponent_params, seeds):
        """-> (score vs the rule-based bot, score vs `opponent_params`); score = (W + D/2) / games, both sides."""
        seeds = list(seeds)
        chunks = [seeds[i::self.n] for i in range(self.n) if seeds[i::self.n]]
        jobs = [("bot", params, None, c) for c in chunks] + [("net", params, opponent_params, c) for c in chunks]
        out = self.pool.map(_play, jobs)
        k = len(chunks)

        def score(part):
            w, d, g = (sum(x[i] for x in part) for i in range(3))
            return (w + 0.5 * d) / g

        return score(out[:k]), score(out[k:])

    def close(self):
        self.pool.terminate()
