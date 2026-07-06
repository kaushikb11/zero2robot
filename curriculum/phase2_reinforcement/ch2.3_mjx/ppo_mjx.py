"""zero2robot 2.3 — 4096 Robots at Once: PPO on MJX.

In ch2.1 you trained ONE cartpole at a time — eight envs stepped in a Python loop.
Real RL scales the other way: thousands of robots stepping together on one
accelerator, so the policy sees millions of transitions per second. That parallel
path for MuJoCo is MJX (MuJoCo-XLA): the same physics, re-expressed in JAX so
`jax.vmap` batches the sim over N worlds and `jax.jit` compiles the whole
rollout+update into one XLA program. This file is ch2.1's PPO — the SAME algorithm
(Gaussian policy, GAE, clipped surrogate, minibatch epochs) — rewritten
functionally and vmapped over `--num_envs` parallel MJX cartpoles.

This is the curriculum's ONE deliberate JAX excursion (like ch1.9's graduation to
LeRobot): the field runs torch for policies and jax for parallel sim, and MJX is
the tool torch cannot honestly replace — there is no parallel MuJoCo in torch.
Read the "JAX vs TORCH" primer region below if you're torch-native.

The lesson is the WALL-CLOCK CLIFF. `--sweep 64,256,1024` measures throughput
(env-steps/sec) as you add parallel envs: it climbs steeply — the parallelism win.
But more envs means each PPO gradient is one big batch, so throughput (data/sec)
and gradient quality (sample efficiency per env step) trade off. MJX's 4096-robot
headline wants a GPU; the CPU-jax free-tier config here is small-but-instructive,
and 4096-on-a-4090 is the Scale Lab.

Run it:    python curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py --seed 0
Cliff:     python .../ppo_mjx.py --sweep 64,256,1024     # throughput vs num_envs
GPU (SL):  python .../ppo_mjx.py --platform gpu --num_envs 4096   # needs jax[cuda]
CI smoke:  python .../ppo_mjx.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch2.3-mjx"))
parser.add_argument("--total_steps", type=int, default=300_000)  # cpu-jax: minutes | smoke: tiny
parser.add_argument("--num_envs", type=int, default=64)     # parallel MJX sims; cpu-jax: 64 | T4: 256 | 4090: 4096
parser.add_argument("--num_steps", type=int, default=128)   # steps per env per rollout -> batch = envs*steps
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--num_minibatches", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE bias/variance knob")
parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO trust region: clip the ratio to 1 +- this")
parser.add_argument("--ent_coef", type=float, default=0.0)
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--eval_envs", type=int, default=64, help="held-out envs for deterministic eval")
parser.add_argument("--seed", type=int, default=0, help="a jax PRNGKey seed; also seeds numpy for logging")
parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu",
                    help="jax backend; cpu is the free-tier default (bitwise-deterministic), gpu is the Scale Lab")
parser.add_argument("--sweep", type=str, default="",
                    help="comma list of num_envs to time throughput at, then exit (the wall-clock cliff)")
parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI; two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false")
args = parser.parse_args()

# JAX reads the backend from the environment at import time, so this must run
# BEFORE `import jax`. cpu keeps the free-tier run deterministic and portable;
# --platform gpu opts into the Scale Lab (needs a jax[cuda] install).
os.environ["JAX_PLATFORMS"] = args.platform
if args.smoke:  # pin everything the CI byte-compare depends on
    args.total_steps, args.num_envs, args.num_steps = 4096, 4, 32
    args.update_epochs, args.num_minibatches, args.eval_envs = 1, 1, 8
    args.platform = "cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import optax  # noqa: E402
from flax import linen as fnn  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402
from mujoco import mjx  # noqa: E402

banner("ch2.3-mjx", device="cuda" if args.platform == "gpu" else "cpu")
args.out.mkdir(parents=True, exist_ok=True)
np.random.seed(args.seed)  # only for logging jitter; the sim/PPO randomness is all PRNGKey
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch2.3-mjx", spawn=False)
    rr.save(str(args.out / "ppo_mjx.rrd"))
# --- endregion ---

# --- region: primer ---
# JAX vs TORCH — the mental-model shift (you spent ~22 chapters in torch):
#   torch: a model OWNS its weights (self.linear.weight), you mutate tensors in
#          place (opt.step()), randomness is a hidden global (manual_seed).
#   jax:   FUNCTIONS ARE PURE. Params are plain data (a pytree of arrays) passed
#          IN and returned OUT — model.apply(params, obs). Nothing mutates: every
#          "update" returns a NEW params pytree. Randomness is EXPLICIT — thread a
#          PRNGKey and split it (key, sub = jax.random.split(key)) at every sample,
#          so the same seed replays bit-for-bit.
#   vmap:  write the env/rollout for ONE world; jax.vmap makes it N worlds for free
#          (a leading batch axis). That is how one device runs 4096 cartpoles.
#   jit:   jax.jit traces the whole rollout+GAE+update into one XLA program and
#          compiles it. First call is slow (compile); every call after is the
#          payoff — that compile-once/run-fast shape is why throughput cliffs.
#   pytree: params, optimizer state, and a batch of MjData are all pytrees; tree_map
#          applies an op to every array leaf at once (we use it for autoreset).
# --- endregion ---

# --- region: env ---
# The env is curriculum/common/envs/cartpole/cartpole.xml loaded into MJX. We do
# NOT reuse CartpoleEnv (that's C-MuJoCo + numpy, imperative); the reset/step/obs
# logic is re-expressed as PURE functions on an mjx.Data so vmap+jit can batch
# them. Constants mirror the env (cartpole_env.py): slider is qpos[0], hinge is
# qpos[1]; reward is +1 per surviving step.
OBS_DIM, ACT_DIM = 5, 1
MAX_STEPS, FRAME_SKIP = 500, 2
ANGLE_LIMIT, CART_LIMIT, RESET_BOUND = 0.2095, 2.4, 0.05

_MJ = mujoco.MjModel.from_xml_path(
    str(Path(__file__).resolve().parents[3] / "curriculum/common/envs/cartpole/cartpole.xml"))
_MX = mjx.put_model(_MJ)  # the XLA-side model; shared (closed over) by every env fn


def env_reset(key):
    """One fresh episode: draw qpos/qvel uniform[-RESET_BOUND, RESET_BOUND] from
    `key`, then mjx.forward to fill derived quantities. Returns an mjx.Data."""
    kq, kv = jax.random.split(key)
    data = mjx.make_data(_MJ).replace(
        qpos=jax.random.uniform(kq, (2,), minval=-RESET_BOUND, maxval=RESET_BOUND),
        qvel=jax.random.uniform(kv, (2,), minval=-RESET_BOUND, maxval=RESET_BOUND),
        ctrl=jnp.zeros(ACT_DIM))
    return mjx.forward(_MX, data)


def env_step(data, action):
    """Apply the clipped force and advance FRAME_SKIP physics steps (50 Hz control
    over 100 Hz physics). lax.scan keeps the inner loop inside XLA."""
    data = data.replace(ctrl=jnp.clip(action, -1.0, 1.0))
    data, _ = jax.lax.scan(lambda d, _: (mjx.step(_MX, d), None), data, None, length=FRAME_SKIP)
    return data


def env_obs(data):
    theta = data.qpos[1]  # hinge angle; 0 = upright. (cos, sin) is seam-free at the top.
    return jnp.array([data.qpos[0], data.qvel[0], jnp.cos(theta), jnp.sin(theta), data.qvel[1]])


def env_terminated(data):  # a REAL failure: pole fell or cart ran off the rail
    return (jnp.abs(data.qpos[1]) > ANGLE_LIMIT) | (jnp.abs(data.qpos[0]) > CART_LIMIT)


# vmap each single-world fn over the leading env axis: THIS is the parallelism.
reset_batch, step_batch = jax.vmap(env_reset), jax.vmap(env_step)
obs_batch, term_batch = jax.vmap(env_obs), jax.vmap(env_terminated)


def tree_select(mask, a, b):
    """Per-env choose whole pytrees a or b (mask True -> a). Autoreset: where an
    episode ended, swap in its fresh-reset Data leaf-by-leaf."""
    return jax.tree_util.tree_map(
        lambda x, y: jnp.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), a, b)
# --- endregion ---

# --- region: model ---
class ActorCritic(fnn.Module):
    """Separate policy and value MLPs (no shared trunk — one less thing to reason
    about) plus a state-INDEPENDENT log-std, exactly like ch2.1's Agent. `apply`
    is pure: params in, (mean, logstd, value) out. Orthogonal init (the quiet PPO
    trick from ch2.1); small actor final gain starts the policy near-still."""
    hidden_dim: int

    @fnn.compact
    def __call__(self, obs):
        ortho = fnn.initializers.orthogonal
        a = fnn.tanh(fnn.Dense(self.hidden_dim, kernel_init=ortho(np.sqrt(2)))(obs))
        a = fnn.tanh(fnn.Dense(self.hidden_dim, kernel_init=ortho(np.sqrt(2)))(a))
        mean = fnn.Dense(ACT_DIM, kernel_init=ortho(0.01))(a)
        logstd = self.param("logstd", fnn.initializers.zeros, (ACT_DIM,))
        c = fnn.tanh(fnn.Dense(self.hidden_dim, kernel_init=ortho(np.sqrt(2)))(obs))
        c = fnn.tanh(fnn.Dense(self.hidden_dim, kernel_init=ortho(np.sqrt(2)))(c))
        return mean, logstd, fnn.Dense(1, kernel_init=ortho(1.0))(c).squeeze(-1)


def gaussian_logprob(action, mean, logstd):
    """Log-prob of `action` under Normal(mean, exp(logstd)), summed over act dims
    (independent Gaussians -> the joint log-prob is the sum)."""
    return (-0.5 * ((action - mean) / jnp.exp(logstd)) ** 2 - logstd - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


def gaussian_entropy(logstd):  # differential entropy of a Gaussian; state-independent here
    return (logstd + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1)
# --- endregion ---

# --- region: rollout ---
def make_rollout(num_steps, num_envs):
    """A jittable rollout: lax.scan over num_steps timesteps, all num_envs stepping
    together. At each step: sample actions from the policy, step every MJX env,
    compute reward/terminated/truncated, autoreset finished envs, record the
    transition. Returns the trajectory (T, N, ...) plus the final carry."""
    def body(carry, _):
        ts, datas, obs, step_count, ep_return, key = carry
        key, ak, rk = jax.random.split(key, 3)
        mean, logstd, value = ts.apply_fn({"params": ts.params}, obs)
        action = mean + jnp.exp(logstd) * jax.random.normal(ak, mean.shape)  # reparameterized sample
        logprob = gaussian_logprob(action, mean, logstd)

        ndatas = step_batch(datas, action)
        nobs = obs_batch(ndatas)
        terminated = term_batch(ndatas)
        step_count = step_count + 1
        truncated = step_count >= MAX_STEPS  # a time limit, NOT a failure -> bootstrap
        done = terminated | truncated
        _, _, next_value = ts.apply_fn({"params": ts.params}, nobs)
        bootstrap = jnp.where(done, next_value, 0.0)  # V(real next state) for GAE on episode ends
        ep_return = ep_return + 1.0  # reward is +1 per surviving step
        finished = jnp.where(done, ep_return, jnp.nan)  # this env's return, if it ended this step

        rdatas = reset_batch(jax.random.split(rk, num_envs))  # autoreset (Brax pattern)
        datas = tree_select(done, rdatas, ndatas)
        obs2 = jnp.where(done[:, None], obs_batch(rdatas), nobs)
        step_count = jnp.where(done, 0, step_count)
        ep_return = jnp.where(done, 0.0, ep_return)
        transition = (obs, action, logprob, value, terminated, done, bootstrap, finished)
        return (ts, datas, obs2, step_count, ep_return, key), transition
    return lambda carry: jax.lax.scan(body, carry, None, length=num_steps)


def compute_gae(traj, last_value, gamma, gae_lambda):
    """GAE over the rollout, walking backward (lax.scan reverse=True). Mirrors
    ch2.1: `1 - terminated` masks the bootstrap so a FALLEN pole contributes no
    future value, while a TRUNCATED episode keeps its stored bootstrap value."""
    _, _, _, values, terminated, done, bootstrap, _ = traj

    def step(carry, t):
        gae, next_value = carry
        next_v = jnp.where(done[t], bootstrap[t], next_value)  # ended step bootstraps from real next state
        delta = 1.0 + gamma * next_v * (1.0 - terminated[t]) - values[t]  # reward is a constant +1
        gae = delta + gamma * gae_lambda * (1.0 - done[t]) * gae
        return (gae, values[t]), gae

    init = (jnp.zeros_like(values[0]), last_value)
    _, advantages = jax.lax.scan(step, init, jnp.arange(values.shape[0]), reverse=True)
    return advantages, advantages + values  # (advantages, returns)
# --- endregion ---

# --- region: update ---
def make_update(args, batch_size, minibatch_size):
    """The PPO update: flatten the rollout, then take update_epochs passes of
    minibatch SGD on the clipped surrogate — the SAME objective as ch2.1, now a
    pure loss function jax differentiates. Returns fn(train_state, flat, key)."""
    def ppo_loss(params, mb, apply_fn):
        obs, action, old_logprob, adv, ret, old_value = mb
        mean, logstd, value = apply_fn({"params": params}, obs)
        ratio = jnp.exp(gaussian_logprob(action, mean, logstd) - old_logprob)  # pi_new/pi_old
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalize advantages per minibatch (trick 1)
        # clipped surrogate: pessimistic (max) of the two -> a too-helpful update clips like a harmful one
        pg = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
        v_clipped = old_value + jnp.clip(value - old_value, -args.clip_coef, args.clip_coef)  # trick 2
        v_loss = 0.5 * jnp.maximum((value - ret) ** 2, (v_clipped - ret) ** 2).mean()
        entropy = gaussian_entropy(logstd).mean()  # trick 3: reward being uncertain, keep exploring
        return pg + args.vf_coef * v_loss - args.ent_coef * entropy, (pg, v_loss, entropy)

    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    def minibatch(ts, mb):
        (loss, aux), grads = grad_fn(ts.params, mb, ts.apply_fn)
        return ts.apply_gradients(grads=grads), (loss, *aux)

    def epoch(carry, _):
        ts, flat, key = carry
        key, pk = jax.random.split(key)
        perm = jax.random.permutation(pk, batch_size)
        shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat)
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, minibatch_size) + x.shape[1:]), shuffled)
        ts, stats = jax.lax.scan(minibatch, ts, minibatches)
        return (ts, flat, key), stats

    def update(ts, flat, key):
        (ts, _, _), stats = jax.lax.scan(epoch, (ts, flat, key), None, length=args.update_epochs)
        return ts, jax.tree_util.tree_map(lambda s: s.mean(), stats)  # (loss, pg, v, entropy)
    return update
# --- endregion ---

# --- region: build ---
def build(args, key):
    """Assemble the train state + parallel env state + one jitted
    update_step(runner) doing rollout -> GAE -> PPO epochs. Rebuilt per num_envs
    so --sweep can time the cliff. The whole step is ONE XLA program."""
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    model = ActorCritic(args.hidden_dim)
    key, mk, rk = jax.random.split(key, 3)
    tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm),  # trick 4: cap the step size
                     optax.adam(args.lr, eps=1e-5))
    train_state = TrainState.create(
        apply_fn=model.apply, params=model.init(mk, jnp.zeros((1, OBS_DIM)))["params"], tx=tx)
    datas = reset_batch(jax.random.split(rk, args.num_envs))
    rollout = make_rollout(args.num_steps, args.num_envs)
    update = make_update(args, batch_size, minibatch_size)

    @jax.jit
    def update_step(runner):
        ts, datas, obs, step_count, ep_return, key = runner
        (ts, datas, obs, step_count, ep_return, key), traj = rollout(runner)
        _, _, last_value = ts.apply_fn({"params": ts.params}, obs)
        advantages, returns = compute_gae(traj, last_value, args.gamma, args.gae_lambda)
        t_obs, t_action, t_logprob, t_value, _, _, _, t_finished = traj
        flat = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]),
            (t_obs, t_action, t_logprob, advantages, returns, t_value))
        key, uk = jax.random.split(key)
        ts, stats = update(ts, flat, uk)
        return (ts, datas, obs, step_count, ep_return, key), (jnp.nanmean(t_finished), stats)

    runner = (train_state, datas, obs_batch(datas),
              jnp.zeros(args.num_envs, dtype=jnp.int32), jnp.zeros(args.num_envs), key)
    return runner, update_step
# --- endregion ---

# --- region: sweep ---
def sweep(args, key):
    """The WALL-CLOCK CLIFF: for each num_envs, compile one update_step, warm it
    up (compilation is one-time), then time 3 steps and report throughput in
    env-steps/sec. Throughput climbs steeply with parallel envs — that climb, and
    where it plateaus on YOUR hardware, is the whole lesson."""
    print(f"\nwall-clock cliff (num_steps={args.num_steps}, platform={args.platform}):")
    print(f"  {'num_envs':>9}  {'env-steps/sec':>14}  {'sec/step':>9}")
    for n in [int(x) for x in args.sweep.split(",") if x.strip()]:
        args.num_envs = n
        runner, update_step = build(args, key)
        for _ in range(2):  # warmup: pay the XLA compile (twice — dtypes settle on call 2)
            runner, _ = update_step(runner)
        jax.block_until_ready(runner)
        t0, reps = time.time(), 3
        for _ in range(reps):
            runner, _ = update_step(runner)
        jax.block_until_ready(runner)
        elapsed = time.time() - t0
        print(f"  {n:>9}  {reps * n * args.num_steps / elapsed:>14,.0f}  {elapsed / reps:>9.3f}")
# --- endregion ---

# --- region: eval ---
def evaluate(apply_fn, params, key, n_envs):
    """Deterministic eval: act with the policy MEAN (no sampling) on held-out
    seeds for MAX_STEPS steps; a dead env stops accruing reward. Mean return ==
    mean steps survived — the number that must climb from random (~10-30) toward
    the cap (500). Jitted scan, cheap even at MAX_STEPS."""
    datas = reset_batch(jax.random.split(key, n_envs))

    def body(carry, _):
        datas, obs, alive, ret = carry
        mean, _, _ = apply_fn({"params": params}, obs)
        ndatas = step_batch(datas, mean)  # deterministic: the mean action, no noise
        ret = ret + alive.astype(jnp.float32)  # +1 for each step still alive
        return (ndatas, obs_batch(ndatas), alive & (~term_batch(ndatas)), ret), None

    init = (datas, obs_batch(datas), jnp.ones(n_envs, dtype=bool), jnp.zeros(n_envs))
    (_, _, _, ret), _ = jax.lax.scan(body, init, None, length=MAX_STEPS)
    return float(jnp.mean(ret))
# --- endregion ---

# --- region: train ---
def main():
    key = jax.random.PRNGKey(args.seed)
    if args.sweep:  # measure the cliff and exit (no training)
        sweep(args, key)
        return

    batch_size = args.num_envs * args.num_steps
    num_iterations = args.total_steps // batch_size
    key, build_key, eval_key = jax.random.split(key, 3)
    runner, update_step = build(args, build_key)

    global_step, mean_return = 0, float("nan")
    for iteration in range(1, num_iterations + 1):
        t0 = time.time()
        runner, (ret, stats) = update_step(runner)
        jax.block_until_ready(runner)
        elapsed = time.time() - t0  # first iter includes the one-time XLA compile
        global_step += batch_size
        mean_return, throughput = float(ret), batch_size / elapsed
        loss, pg, v_loss, entropy = (float(s) for s in stats)
        if args.rerun:
            rr.set_time("global_step", sequence=global_step)
            rr.log("charts/episodic_return", rr.Scalars([mean_return]))  # the learning curve
            rr.log("charts/throughput_env_steps_per_sec", rr.Scalars([throughput]))  # the cliff, live
            rr.log("losses/policy", rr.Scalars([pg]))
            rr.log("losses/value", rr.Scalars([v_loss]))
        if iteration % 5 == 0 or iteration == num_iterations:
            print(f"iter {iteration:3d}/{num_iterations}  step {global_step:7d}  "
                  f"return {mean_return:6.1f}  {throughput:9,.0f} env-steps/s  pg {pg:+.3f}  v {v_loss:.2f}")

    ts = runner[0]
    mean_eval = evaluate(ts.apply_fn, ts.params, eval_key, args.eval_envs)
    print(f"eval: mean return {mean_eval:.1f} over {args.eval_envs} envs (random ~10-30, cap {MAX_STEPS})")
    metrics = {
        "mean_eval_return": round(mean_eval, 6),
        "mean_train_return_final": round(mean_return, 6),
        "num_envs": args.num_envs, "num_iterations": num_iterations, "num_steps": args.num_steps,
        "seed": args.seed, "smoke": bool(args.smoke), "total_steps": global_step,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"metrics: {args.out / 'metrics.json'}")
    if args.rerun:
        print(f"recording: {args.out / 'ppo_mjx.rrd'} — open it with: rerun {args.out / 'ppo_mjx.rrd'}")


if __name__ == "__main__":
    main()
# --- endregion ---
