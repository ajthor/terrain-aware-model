import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- Helper for plotting histogram + normal curve ---
def plot_hist(ax, data, title, color, units=None, save_text=None):
    mu, sigma = norm.fit(data)
    x = np.linspace(min(data), max(data), 1000)
    pdf = norm.pdf(x, mu, sigma)
    ax.hist(data, bins=30, density=True, alpha=0.6, color=color, edgecolor='black')
    ax.plot(x, pdf, 'r-', linewidth=2, label=f'N({mu:.5f}, {sigma:.5f}²)')
    ax.set_title(title)
    ax.set_xlabel(f"Value ({units})")
    ax.set_ylabel("Density")
    ax.legend()
    if save_text:
        print(save_text.format(mu=mu, sigma=sigma, units=units or ""))


def main():

    print("\n=================== INSPECTING RAW DATA ===================\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", type=str, default='')
    parser.add_argument("--transmission", type=str, default='')
    parser.add_argument("--type", type=str, default='train')
    parser.add_argument("--animation", type=bool, default=False)
    args = parser.parse_args()

    save_path = f"function_encoder_beamng/data/{args.transmission}/{args.type}/{args.terrain}/inspections/raw_data"
    os.makedirs(save_path, exist_ok=True)

    # Load data
    path = f"function_encoder_beamng/data/{args.transmission}/{args.type}/{args.terrain}"
    state_df = pd.read_csv(f"{path}/state_sensor.csv")
    imu_df = pd.read_csv(f"{path}/imu_angular_velocity.csv")

    # Extract data
    time = state_df["time"].values
    pos  = state_df[["pos_x", "pos_y", "pos_z"]].values
    dirs = state_df[["dir_x", "dir_y", "dir_z"]].values
    ups  = state_df[["up_x", "up_y", "up_z"]].values
    vel  = state_df[["vel_x", "vel_y", "vel_z"]].values
    ang_vel = imu_df[["ang_vel_x", "ang_vel_y", "ang_vel_z"]].values




    # --- Check terrain bounds ---
    outside = np.where((pos[:, 0] < 0) | (pos[:, 0] > 4096) | (pos[:, 1] < 0) | (pos[:, 1] > 4096))[0]
    if len(outside) == 0:
        print("✅ Vehicle stayed within terrain bounds.")
    else:
        print(f"❌ Vehicle drove outside bounds at {len(outside)} time stamps.")




    # --- z-Axis position & velocity ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_hist(axs[0], pos[:, 2], "Global Position Z", 'skyblue', "m",
            "\nZ Position = {mu:.5f} ± {sigma:.5f} {units}.")
    plot_hist(axs[1], vel[:, 2], "Global Velocity Z", 'lightgreen', "m/s",
            "Z Velocity = {mu:.5f} ± {sigma:.5f} {units}. (Expecting 0 m/s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{save_path}/z_stats.png", bbox_inches="tight", dpi=300)
    plt.close()




    # --- Orthogonality (ups, dirs) with Z ---
    Z = np.array([0, 0, 1])
    upDotZ, dirDotZ = ups @ Z, dirs @ Z

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_hist(axs[0], upDotZ, "ups @ Z", 'skyblue', "unitless",
            "\nUp @ Z = {mu:.5f} ± {sigma:.5f}. (Expecting 1)")
    plot_hist(axs[1], dirDotZ, "dirs @ Z", 'lightgreen', "unitless",
            "Dir @ Z = {mu:.5f} ± {sigma:.5f}. (Expecting 0)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{save_path}/dot_products.png", bbox_inches="tight", dpi=300)
    plt.close()




    # --- Angular velocities (pitch & roll) ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_hist(axs[0], ang_vel[:, 0] * 180 / np.pi, "X Angular Velocity",
            'skyblue', "deg/s", "\nX Angular Velocity = {mu:.4f} ± {sigma:.4f} {units}. (Expecting 0 deg/s)")
    plot_hist(axs[1], ang_vel[:, 1] * 180 / np.pi, "Y Angular Velocity",
            'lightgreen', "deg/s", "Y Angular Velocity = {mu:.4f} ± {sigma:.4f} {units}. (Expecting 0 deg/s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{save_path}/ang_vels.png", bbox_inches="tight", dpi=300)
    plt.close()




    # --- Setup subplots (rows = states, cols = components) ---
    fig, axs = plt.subplots(5, 3, figsize=(12, 8), sharex=True)

    state_names = ["Position", "Direction", "Up", "Velocity", "Ang Vel"]
    colors = ["tab:blue", "tab:green", "tab:red"]
    labels = ["x", "y", "z"]

    # Data groups for iteration
    groups = [pos, dirs, ups, vel, ang_vel]

    for row, (state, name) in enumerate(zip(groups, state_names)):
        for col in range(3):
            ax = axs[row, col]
            ax.plot(time, state[:, col], color=colors[col], lw=1.5)
            # ax.scatter(time, state[:, col], color=colors[col])
            if row == 0:
                ax.set_title(f"{labels[col].upper()} Component", fontsize=12)
            if col == 0:
                ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)

    # Common X label
    for ax in axs[-1, :]:
        ax.set_xlabel("Time [s]")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{save_path}/raw_states.png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"\nSaved plots to {save_path}.")




    if args.animation:

        # --- Plot pose in 3D space ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set equal aspect ratio for better visualization
        max_range = np.ptp(pos, axis=0).max() / 2.0
        mid = np.mean(pos, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        # --- Initialize trajectory line ---
        line, = ax.plot([], [], [], lw=2, color="gray")

        # --- Update function for animation ---
        def update(frame):
            ax.collections.clear()  # clear old arrows

            # Update trajectory line
            line.set_data(pos[:frame, 0], pos[:frame, 1])
            line.set_3d_properties(pos[:frame, 2])

            # Current position and vectors
            p = pos[frame]
            v = vel[frame]
            d = dirs[frame]
            u = ups[frame]

            # Global up vector (Z-axis)
            z = np.array([0, 0, 1])

            # Plot vectors (velocity=red, direction=green, up=magenta)
            ax.quiver(p[0], p[1], p[2], v[0], v[1], v[2],
                    color="red", length=30, normalize=False)
            ax.quiver(p[0], p[1], p[2], d[0], d[1], d[2],
                    color="green", length=500, normalize=False)
            ax.quiver(p[0], p[1], p[2], u[0], u[1], u[2],
                    color="green", length=500, normalize=False)
            ax.quiver(p[0], p[1], p[2], z[0], z[1], z[2],
                    color="blue", length=500, normalize=False)

            ax.set_title(f"Time = {time[frame]:.2f}s")
            return line,

        # --- Create animation ---
        ani = FuncAnimation(fig, update, frames=len(pos), interval=5, blit=False)

        # --- Save MP4 ---
        os.makedirs(save_path, exist_ok=True)
        mp4_path = f"{save_path}/trajectory.mp4"

        writer = FFMpegWriter(fps=30, bitrate=1800)
        ani.save(mp4_path, writer=writer)
        print(f"Saved trajectory animation to {mp4_path}.")




        # --- Setup 3D figure ---
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set equal aspect ratio for better visualization
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)

        # --- Initialize trajectory line ---
        line, = ax.plot([], [], [], lw=2, color="gray")

        # --- Update function for animation ---
        def update(frame):
            ax.collections.clear()  # clear old arrows

            # Current position and vectors
            v = vel[frame]
            d = dirs[frame]
            u = ups[frame]

            # Global up vector (Z-axis)
            z = np.array([0, 0, 1])

            # Plot vectors (velocity=red, direction=green, up=magenta)
            ax.quiver(0, 0, 0, v[0], v[1], v[2],
                    color="red", length=0.1, normalize=False)
            ax.quiver(0, 0, 0, d[0], d[1], d[2],
                    color="green", length=5, normalize=False)
            ax.quiver(0, 0, 0, u[0], u[1], u[2],
                    color="green", length=5, normalize=False)
            ax.quiver(0, 0, 0, z[0], z[1], z[2],
                    color="blue", length=5, normalize=False)

            ax.set_title(f"Time = {time[frame]:.2f}s")
            return line,

        # --- Create animation ---
        ani = FuncAnimation(fig, update, frames=len(vel), interval=5, blit=False)

        # --- Save MP4 ---
        os.makedirs(save_path, exist_ok=True)
        mp4_path = f"{save_path}/orientation.mp4"

        writer = FFMpegWriter(fps=30, bitrate=1800)
        ani.save(mp4_path, writer=writer)
        print(f"Saved orientation animation to {mp4_path}.")

    else:
        print("Skipping animations.")


if __name__ == '__main__':
    main()