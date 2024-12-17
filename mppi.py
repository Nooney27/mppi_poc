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
import numpy as np
from typing import Tuple
import math


class MPPI():
    def __init__(
            self,
            m: float = 1.0, 
            l: float = 1.0, 
            max_torque: float = 2.0, 
            max_speed: float = 8.0, 
            dt: float = 0.05,
            horizon: int = 30,
            num_samples: int = 1000,
            param_exploration: float = 0.01,
            param_lambda: float = 1.0,
            param_alpha: float = 0.1,
            sigma: float = 1.0,
            stage_cost_weight: np.ndarray = np.array([1.0, 0.1]),
            terminal_cost_weight: np.ndarray = np.array([1.0, 0.1]),
    ) -> None:
        # initialize parameters
        # mppi:
        self.dim_u = 1
        self.T = horizon # prediction horizon
        self.K = num_samples
        self.param_exploration = param_exploration # exploration parameter constant of mppi
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.sigma = sigma # standard deviation of noise
        self.param_gamma = self.param_lambda * (1.0 - self.param_alpha)
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight

        self.u_prev = np.zeros(self.T)

        # environment parameters: Pendulum
        self.g = 9.81
        self.m = m
        self.l = l
        self.dt = dt
        self.max_torque = max_torque
        self.max_speed = max_speed
 
    def control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        u = self.u_prev
        x0 = observed_x
        S = np.zeros(self.K) # state cost list
        epsilon = self.calc_epsilon(self.sigma, self.K, self.T) 
        trajectories = []

        for k in range(self.K):
            v = np.zeros(self.K)
            x = x0
            traj = [x.copy()]

            for t in range(1, self.T+1):
                # control input with noise
                if k < (1.0 - self.param_exploration) * self.K:
                    v[t-1] = u[t-1] + epsilon[k, t-1] 
                else:
                    v[t-1] = epsilon[k, t-1]
                
                x = self.update(x, self.clamp_input(v[t-1]))
                traj.append(x.copy())
                S[k] += self.stage_cost(x) + self.param_gamma * u[t-1] * (1.0/self.sigma) * v[t-1]
            
            S[k] += self.terminal_cost(x)
            trajectories.append(np.array(traj))

        w = self.calc_weight(S)
        w_epsilon = np.zeros(self.T)
        for t in range(self.T):
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]
        
        # moving average filter for smoothing input sequence
        w_epsilon = self.filter(xx=w_epsilon, window_size=5)

        u = u + w_epsilon

        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        min_cost = np.min(S)

        return u[0], u, trajectories, min_cost
    
    def update(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        # calculate next state of the system
        theta, theta_dot = x_t[0], x_t[1]
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        torque = v_t
        new_theta_dot = theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * torque) * dt
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * dt

        x_t_next = np.array([new_theta, new_theta_dot])
        return x_t_next
    
    def stage_cost(self, x_t: np.ndarray) -> float:
        theta, theta_dot = x_t[0], x_t[1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi  # normalize theta to [-pi, pi]

        # Penalize both angular position and angular velocity
        stage_cost = self.stage_cost_weight[0] * theta**2 + self.stage_cost_weight[1] * theta_dot**2
        return stage_cost

    
    def terminal_cost(self, x_t: np.ndarray) -> float:
        # terminal cost function
        theta, theta_dot = x_t[0], x_t[1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        terminal_cost = self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2
        return terminal_cost
    
    def calc_epsilon(self, sigma: float, K: int, T: int) -> np.ndarray:
        # sample epsilon
        epsilon = np.random.normal(0.0, sigma, (K, T))
        return epsilon
    
    def calc_weight(self, S: np.ndarray) -> np.ndarray:
        # calculate wieght for each sample
        w = np.zeros(self.K)

        rho = S.min()
        eta = 0.0
        for k in range(self.K):
            eta += np.exp(-1.0/self.param_lambda * (S[k] - rho))
        
        for k in range(self.K):
            w[k] = (1.0 / eta) * np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        return w
    
    def filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        b = np.ones(window_size)/window_size
        xx_mean = np.convolve(xx, b, mode="same")
        n_conv = math.ceil(window_size/2)
        xx_mean[0] *= window_size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= window_size/(i+n_conv)
            xx_mean[-i] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean
    
    def clamp_input(self, v: float) -> float:
        # clip control input
        v = np.clip(v, -self.max_torque, self.max_torque)
        return v
    
