import numpy as np
from environment_jax import Pendulum
from mppi_jax import MPPI
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp
#print("JAX is installed. Backend:", jax.devices()[0])


# Settings:
delta_t = 0.05
sim_steps = 200
print(f"[INFO] delta_t : {delta_t:.2f}[s] , sim_steps : {sim_steps}[steps], total_sim_time : {delta_t*sim_steps:.2f}[s]")

# data collection
theta_list = []
theta_dot_list = []
torque_list = []
time_list = []
sampled_trajectories = []
cost_list = []
runtime_list = []
applied_trajectory = []


# initialize a pendulum as a control target
pendulum = Pendulum(
    m = 1.0,
    l = 1.0,
    max_torque = 2.0,
    max_speed = 8.0,
    dt = delta_t,
)
pendulum.reset(
    init_state = np.array([np.pi, 0.0]), # [theta(rad), theta_dot(rad/s)]
)

# initialize a mppi controller for the pendulum
mppi = MPPI(
    dt = delta_t,
    m = 1.0,
    l = 1.0,
    max_torque = 2.0,
    max_speed = 8.0,
    horizon = 20,
    num_samples= 1500,
    param_lambda = 0.5,
    sigma = 1.0,
    stage_cost_weight    = np.array([1.0, 0.1]), # weight for [theta, theta_dot]
    terminal_cost_weight = 10.0 * np.array([1.0, 0.5]), # weight for [theta, theta_dot]
)

# simulation loop
for i in range(sim_steps):
    start_time = time.time()

    current_state = pendulum.get_state()
    
    input_torque, input_torque_sequence, trajectories, min_cost = mppi.control_input(observed_x=current_state)
    
    runtime = time.time() - start_time
    runtime_list.append(runtime)
   
    # print current state and input torque
    print(f"Time: {i*delta_t:>2.2f}[s], theta={current_state[0]:>+3.3f}[rad], theta_dot={current_state[1]:>+3.3f}[rad/s], input torque={input_torque:>+3.2f}[Nm]", end="")
    print(", Upward position reached" if abs(current_state[0]) < 0.1 and abs(current_state[1] < 0.1) else "")
                                                  
    # Store data
    theta_list.append(current_state[0])
    theta_dot_list.append(current_state[1])
    torque_list.append(input_torque)
    sampled_trajectories.append(trajectories)
    cost_list.append(min_cost)
    time_list.append(i * delta_t)
    applied_trajectory.append(current_state.copy)

    pendulum.update(u=[input_torque], dt=delta_t)



pendulum.save_animation(filename="pendulum_jax.mp4", interval=delta_t*1000, movie_writer="ffmpeg")
pendulum.save_animation(filename="pendulum_jax.gif", interval=delta_t*1000, movie_writer="pillow")


# Plot MPPI Sampled Trajectories
plt.figure(figsize=(10, 6))
for k, traj_set in enumerate(sampled_trajectories[::10]):  # Plot every 10th step
    for traj in traj_set:
        plt.plot(range(len(traj)), traj[:, 0], alpha=0.1, color="gray")  # theta trajectory
plt.title("MPPI Sampled Trajectories")
plt.xlabel("Time Step")
plt.ylabel("Theta (rad)")
plt.show()

frequencies = [1 / rt for rt in runtime_list]
print(f"Average Frequency: {np.mean(frequencies):.2f} Hz")


plt.figure(figsize=(14, 8))

# Theta evolution
plt.subplot(4, 1, 1)
plt.plot(time_list, theta_list, label="Theta (rad)")
plt.title("Pendulum State Evolution and Costs")
plt.ylabel("Theta (rad)")
plt.legend()

# Theta_dot evolution
plt.subplot(4, 1, 2)
plt.plot(time_list, theta_dot_list, label="Theta_dot (rad/s)", color="orange")
plt.ylabel("Theta_dot (rad/s)")
plt.legend()

# Input Torque
plt.subplot(4, 1, 3)
plt.plot(time_list, torque_list, label="Input Torque (Nm)", color="green")
plt.ylabel("Torque (Nm)")
plt.legend()

# Cost Evolution
plt.subplot(4, 1, 4)
plt.plot(time_list, cost_list, label="Minimum Cost", color="red")
plt.xlabel("Time (s)")
plt.ylabel("Cost")
plt.legend()

plt.tight_layout()
plt.show()
