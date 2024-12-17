'''
MPPI: Model Predictive Path Integral Control

1) Randomly sample control input sequence

U = mean input sequence ("optimal" input sequence)
V = randomly sampled input sequence
v_t = u_t + epsilon_t

2) predict future states and cost evaluations for each sample
x = state
v = sampled control input
x_t+1 = f(x_t, v_t)

the cost for a sampled sequence S(V, x_0)

3) weight for each sample sequence
w = exp(-1/lambda * J)

4) update law: optimal control input

u_t = u_t + sum(w_t)*epsilon_t

'''
import jax
import jax.numpy as jnp
from typing import Tuple

class MPPI:
    def __init__(
        self,
        m: float = 1.0, 
        l: float = 1.0, 
        max_torque: float = 2.0, 
        max_speed: float = 8.0, 
        dt: float = 0.05,
        horizon: int = 30,
        num_samples: int = 1000,
        sigma: float = 1.0,
        param_lambda: float = 1.0,
        stage_cost_weight: jnp.ndarray = jnp.array([1.0, 0.1]),
        terminal_cost_weight: jnp.ndarray = jnp.array([10.0, 1.0])
    ):
        self.m = m
        self.l = l
        self.g = 9.81
        self.dt = dt
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.T = horizon
        self.K = num_samples
        self.sigma = sigma
        self.lambda_param = param_lambda
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight
        self.u_prev = jnp.zeros(self.T)

    @jax.jit
    def update_state(self, x, u):
        """Dynamics of the pendulum (JAX-compatible)."""
        theta, theta_dot = x
        new_theta_dot = theta_dot + (3 * self.g / (2 * self.l) * jnp.sin(theta) + 3.0 / (self.m * self.l**2) * u) * self.dt
        new_theta_dot = jnp.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * self.dt
        return jnp.array([new_theta, new_theta_dot])

    def sample_noise(self, key):
        """Sample Gaussian noise for control inputs."""
        return jax.random.normal(key, (self.K, self.T)) * self.sigma

    @staticmethod
    @jax.jit
    def simulate_trajectory(x0, u_seq, noise, g, m, l, dt, max_speed):
        """Simulate trajectories given initial state, control sequence, and noise."""
        def step(x, u_noise):
            u = u_noise
            theta, theta_dot = x
            new_theta_dot = theta_dot + (3 * g / (2 * l) * jnp.sin(theta) + 3.0 / (m * l**2) * u) * dt
            new_theta_dot = jnp.clip(new_theta_dot, -max_speed, max_speed)
            new_theta = theta + new_theta_dot * dt
            return jnp.array([new_theta, new_theta_dot]), None

        u_noisy = u_seq + noise
        x_trajectory, _ = jax.lax.scan(step, x0, u_noisy)
        return x_trajectory


    @staticmethod
    @jax.jit
    def compute_costs(trajectories, stage_cost_weight, terminal_cost_weight):
        """Compute stage and terminal costs for all trajectories."""
        theta, theta_dot = trajectories[..., 0], trajectories[..., 1]
        stage_costs = stage_cost_weight[0] * theta**2 + stage_cost_weight[1] * theta_dot**2
        terminal_costs = terminal_cost_weight[0] * theta[-1]**2 + terminal_cost_weight[1] * theta_dot[-1]**2
        total_costs = jnp.sum(stage_costs, axis=1) + terminal_costs
        return total_costs


    @jax.jit
    def compute_weights(self, costs):
        """Compute weights for sampled trajectories."""
        min_cost = jnp.min(costs)
        weights = jnp.exp(-1.0 / self.lambda_param * (costs - min_cost))
        return weights / jnp.sum(weights)

    @jax.jit
    def update_control(self, u_prev, weights, noise):
        """Update the control sequence using weighted noise."""
        weighted_noise = jnp.sum(weights[:, None] * noise, axis=0)
        return u_prev + weighted_noise

    def control_input(self, observed_x: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """Main MPPI loop to compute the optimal control input."""
        key = jax.random.PRNGKey(0)
        noise = self.sample_noise(key)
        u = self.u_prev

        # Simulate all trajectories in parallel
        x0 = observed_x
        u_seq = jnp.tile(u, (self.K, 1))  # Repeat u_prev for all samples
        simulate_vmap = jax.vmap(self.simulate_trajectory, in_axes=(None, 0, 0, None, None, None, None, None))
        trajectories = simulate_vmap(x0, u_seq, noise, self.g, self.m, self.l, self.dt, self.max_speed)

        # Compute costs and weights
        costs = self.compute_costs(trajectories, self.stage_cost_weight, self.terminal_cost_weight)
        weights = self.compute_weights(costs)

        # Update control inputs
        u = self.update_control(u, weights, noise)

        # Update u_prev for the next timestep
        self.u_prev = jnp.concatenate((u[1:], u[-1:]))  # Shift and hold last control

        # Return the first control input to apply
        return u[0], u

