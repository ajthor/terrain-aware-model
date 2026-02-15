import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from inspect_raw_data import plot_hist

print("\n================ INSPECTING PROCESSED DATA ================\n")

parser = argparse.ArgumentParser()
parser.add_argument("--terrain", type=str, default='')
parser.add_argument("--transmission", type=str, default='')
parser.add_argument("--type", type=str, default='train')
args = parser.parse_args()

save_path = f"function_encoder_beamng/data/{args.transmission}/{args.type}/{args.terrain}/inspections/processed_data"
os.makedirs(save_path, exist_ok=True)

# Load data
base_dir = "function_encoder_beamng/data_split/seed_0"
data_dir = os.path.join(base_dir, args.transmission, args.terrain)
input_path = os.path.join(data_dir, "train_input.csv")
target_path = os.path.join(data_dir, "train_target.csv")

def _input_headers(num_cols):
    base = ["Time", "x", "y", "yaw", "vx", "vy", "vyaw", "brake", "parking brake", "steer", "throttle"]
    if num_cols == len(base) + 1:
        base.insert(7, "engine_rpm")
    if num_cols <= len(base):
        return base[:num_cols]
    return base + [f"feat_{i}" for i in range(num_cols - len(base))]

def _target_headers(num_cols):
    base = ["Time", "del_x", "del_y", "del_yaw", "del_vx", "del_vy", "del_vyaw"]
    if num_cols == len(base) + 1:
        base.append("del_rpm")
    if num_cols <= len(base):
        return base[:num_cols]
    return base + [f"feat_{i}" for i in range(num_cols - len(base))]

train_input = pd.read_csv(input_path, header=None)
train_target = pd.read_csv(target_path, header=None)
train_input.columns = _input_headers(train_input.shape[1])
train_target.columns = _target_headers(train_target.shape[1])




# --- Input–Target Scatter Grid ---
input_cols = ["vx", "vy", "vyaw", "brake", "parking brake", "steer", "throttle"]
target_cols = ["del_x", "del_y", "del_yaw", "del_vx", "del_vy", "del_vyaw"]

fig, axes = plt.subplots(len(target_cols), len(input_cols), figsize=(16, 12), sharex=False, sharey=False)

for i, tcol in enumerate(target_cols):
    for j, icol in enumerate(input_cols):
        ax = axes[i, j]

        ax.scatter(
            train_input[icol],
            train_target[tcol],
            s=3,
            alpha=0.5,
        )

        if i == len(target_cols) - 1:
            ax.set_xlabel(icol, fontsize=8)
        else:
            ax.set_xticklabels([])

        if j == 0:
            ax.set_ylabel(tcol, fontsize=8)
        else:
            ax.set_yticklabels([])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{save_path}/targets_vs_inputs.png", bbox_inches="tight", dpi=300)
plt.close()




# --- Velocity–Control Scatter Grid ---
velocity_cols = ["vx", "vy", "vyaw"]
control_cols = ["brake", "parking brake", "steer", "throttle"]

fig2, axes2 = plt.subplots(len(velocity_cols), len(control_cols), figsize=(12, 8), sharex=False, sharey=False)

for i, vcol in enumerate(velocity_cols):
    for j, ccol in enumerate(control_cols):
        ax = axes2[i, j]
        ax.scatter(train_input[ccol], train_input[vcol], s=3, alpha=0.5)
        if i == len(velocity_cols) - 1:
            ax.set_xlabel(ccol, fontsize=8)
        else:
            ax.set_xticklabels([])
        if j == 0:
            ax.set_ylabel(vcol, fontsize=8)
        else:
            ax.set_yticklabels([])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{save_path}/inputs_vs_inputs.png", bbox_inches="tight", dpi=300)
plt.close()



# --- Target-Target Scatter Grid ---
target_cols = ["del_x", "del_y", "del_yaw", "del_vx", "del_vy", "del_vyaw"]

fig, axes = plt.subplots(len(target_cols), len(target_cols), figsize=(16, 12), sharex=False, sharey=False)

for i, tcol in enumerate(target_cols):
    for j, icol in enumerate(target_cols):
        ax = axes[i, j]

        ax.scatter(
            train_target[icol],
            train_target[tcol],
            s=3,
            alpha=0.5,
        )

        if i == len(target_cols) - 1:
            ax.set_xlabel(icol, fontsize=8)
        else:
            ax.set_xticklabels([])

        if j == 0:
            ax.set_ylabel(tcol, fontsize=8)
        else:
            ax.set_yticklabels([])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{save_path}/targets_vs_targets.png", bbox_inches="tight", dpi=300)
plt.close()





# --- del_x vs vx colored by Time ---
fig4, ax4 = plt.subplots(figsize=(8, 6))
sc = ax4.scatter(
    train_input["vx"],
    train_target["del_x"],
    c=train_input["Time"],
    cmap="plasma",
    s=8,
    alpha=0.8
)
ax4.set_xlabel("vx")
ax4.set_ylabel("\u0394x")
ax4.set_title(f"\u0394x vs vx")
cb = plt.colorbar(sc, ax=ax4)
cb.set_label("Time")

plt.tight_layout()
plt.savefig(f"{save_path}/del_x_vs_vx.png", bbox_inches="tight", dpi=300)
plt.close()




# --- Histogram of del Time ---
fig, ax = plt.subplots(figsize=(6, 6))
del_time = train_target["Time"] - train_input["Time"]
plot_hist(ax, del_time, "Target Time - Train Time", 'skyblue', "s",
           "\u0394t = {mu:.5f} ± {sigma:.5f} {units}. (Expecting 0.05 s)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{save_path}/del_time.png", bbox_inches="tight", dpi=300)
plt.close()
