import numpy as np
import torch


def process_data(states, angvels, electrics, include_rpm: bool = False):
    """
    Process data for training dynamics models.
    
    INPUTS:
        states:

            time 

            x position in the global frame
            y position in the global frame
            z position in the global frame

            x coordinate of orientation unit vector in global frame
            y coordinate of orientation unit vector in global frame
            z coordinate of orientation unit vector in global frame

            x coordinate of up unit vector in global frame
            y coordinate of up unit vector in global frame
            z coordinate of up unit vector in global frame

            x velocity in the inertial frame
            y velocity in the inertial frame
            z velocity in the inertial frame

        angvels:

            time

            ang_vel_x in the body frame
            ang_vel_y in the body frame
            ang_vel_z in the body frame

        controls:
            
            brake
            parking brake
            steering
            throttle

        engine_rpm (optional):
            engine RPM from electrics (when include_rpm=True)
    """

    # Align lengths (drop extra tail samples if one stream ends early).
    n_states = states.shape[0]
    n_angvels = angvels.shape[0]
    n_electrics = len(electrics)
    min_len = min(n_states, n_angvels, n_electrics)
    if (n_states, n_angvels, n_electrics) != (min_len, min_len, min_len):
        print(
            f"⚠️ Length mismatch: states={n_states}, angvels={n_angvels}, electrics={n_electrics}. "
            f"Trimming to {min_len}."
        )
        states = states[:min_len]
        angvels = angvels[:min_len]
        electrics = electrics.iloc[:min_len].reset_index(drop=True)

    # Extract the control inputs
    cols = ["brake_input", "parkingbrake_input", "steering_input", "throttle_input"]
    controls = electrics[cols].to_numpy(dtype=np.float64)

    rpm = None
    if include_rpm:
        if "engine_rpm" not in electrics.columns:
            raise ValueError("include_rpm=True but 'engine_rpm' column not found in electrics.")
        rpm = electrics["engine_rpm"].to_numpy(dtype=np.float64)

    # # Extract the wheel speed state
    # cols = ["wheelspeed"]
    # wheelspeed = electrics[cols].to_numpy(dtype=np.float64)

    # Convert to torch tensors
    states = torch.tensor(states, dtype=torch.float64)
    controls = torch.tensor(controls, dtype=torch.float64)
    angvels = torch.tensor(angvels, dtype=torch.float64)
    if rpm is not None:
        if rpm.shape[0] != states.shape[0]:
            raise ValueError("engine_rpm length does not match state length.")
        rpm = torch.tensor(rpm, dtype=torch.float64).unsqueeze(-1)
    # wheelspeed = torch.tensor(wheelspeed, dtype=torch.float64)

    # Concatenate the angular velocities with the other states
    # states = torch.cat((states, angvels[:,1:], wheelspeed), dim=1)
    states = torch.cat((states, angvels[:,1:]), dim=1)

    # Subtract off the initial time.
    states[:, 0] = states[:, 0] - states[0, 0]

    """ Investigate the change in time. """
    # del_t = states[1:,0] - states[:-1,0]
    # import matplotlib.pyplot as plt
    # plt.scatter(states[:-1, 0], del_t)
    # NOTE that the average is 0.05 seconds. 

    """ Investigate the controls. """
    # import matplotlib.pyplot as plt
    # plt.plot(controls[:, 106], controls[:, 11])
    # plt.show()

    # Split into inputs and targets
    inputs = states[:-1].clone()
    targets = states[1:].clone()

    # Drop the final control
    controls = controls[:-1].clone()
    if rpm is not None:
        rpm_inputs = rpm[:-1].clone()
        rpm_targets = rpm[1:].clone()

    # Get the change in position w.r.t the initial body frame.
    targets = pos_inertial_to_body(refs=inputs, data=targets)

    # Get the initial linear velocity w.r.t the initial body frame.
    inputs = vel_inertial_to_body(refs=inputs, data=inputs)

    # Get the final linear velocity w.r.t the initial body frame. 
    targets = vel_inertial_to_body(refs=inputs, data=targets)

    # Calculate the yaw angles from tan
    input_yaws = torch.atan2(inputs[:,5], inputs[:,4])
    target_yaws = torch.atan2(targets[:,5], targets[:,4])

    # Unwrap the yaw angles
    input_yaws = np.unwrap(input_yaws)
    target_yaws = np.unwrap(target_yaws)
    
    # Get the change in yaws
    target_yaws = torch.from_numpy(target_yaws - input_yaws)

    # Replace the cos data with yaw data
    inputs[:,4] = torch.from_numpy(input_yaws)
    targets[:,4] = target_yaws

    # Get the change in linear velocity and ang velocity
    targets[:, 10:16] = targets[:, 10:16] - inputs[:, 10:16]

    # Zero out the position and heading.
    inputs[:, 1:6] = 0

    # Only keep the relevant states.
    inputs = inputs[:, [0, 1, 2, 4, 10, 11, 15]]
    targets = targets[:, [0, 1, 2, 4, 10, 11, 15]]
    if rpm is not None:
        delta_rpm = rpm_targets - rpm_inputs
        inputs = torch.cat((inputs, rpm_inputs), dim=1)
        targets = torch.cat((targets, delta_rpm), dim=1)

    # Concatenate inputs and cmd_vel
    inputs = torch.cat((inputs, controls), dim=1)

    return inputs, targets


def pos_inertial_to_body(refs, data):

    """ Returns the change in position w.r.t the input body frame. """

    x = refs[:,4]
    y = refs[:,5]

    c = x / torch.sqrt(x**2 + y**2) # approximate cosine of yaw angle
    s = y / torch.sqrt(x**2 + y**2) # approximate sine of yaw angle

    R = torch.stack(
        [
            torch.stack([c, s], dim=1),
            torch.stack([-s, c], dim=1),
        ],
        dim=1,
    )

    # Apply the rotation matrix to the odom data
    data[:, 1:3] = torch.bmm(R, (data[:, 1:3] - refs[:, 1:3]).unsqueeze(-1)).squeeze(-1)

    return data


def vel_body_to_inertial(refs, data):

    c = refs[:,3]
    s = refs[:,4]

    R = torch.stack(
        [
            torch.stack([c, -s], dim=1),
            torch.stack([s, c], dim=1),
        ],
        dim=1,
    )

    # Apply the rotation matrix to the odom velocity data
    data[:, 5:7] = torch.bmm(R, data[:, 5:7].unsqueeze(-1)).squeeze(-1)

    return data


def vel_inertial_to_body(refs, data):

    x = refs[:,4]
    y = refs[:,5]

    c = x / torch.sqrt(x**2 + y**2) # approximate cosine of yaw angle
    s = y / torch.sqrt(x**2 + y**2) # approximate sine of yaw angle

    R = torch.stack(
        [
            torch.stack([c, s], dim=1),
            torch.stack([-s, c], dim=1),
        ],
        dim=1,
    )

    # Apply the rotation matrix to the odom velocity data
    data[:, 10:12] = torch.bmm(R, data[:, 10:12].unsqueeze(-1)).squeeze(-1)

    return data
