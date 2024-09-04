import math

time_step = 0.1
time_horizon = 3.3

print("using time_step:", time_step)
inv_time_step = 1.0 / time_step
inv_time_horizon = 1.0 / time_horizon

velocity = tuple((0, 0.1))
other_velocity = tuple((0, -0.1))
pos = tuple((0, 0))
other_pos = tuple((0, 0.008))
combined_radius = 0.108

# Result from simulator

# Collision agent id: 13
# Collision agent position: (0.155286,1.02437)
# Collision relative velocity: (-0.118345,-0.102089)
# Collision relative position: (0.0768028,0.0705177)
# calculated w: (-0.886374,-0.807266)
# calculated u: (0.0878987,0.0800539)

# using time_step: 0.1
# relative_velocity: (-0.118345, -0.102089)
# relative_position: (0.0768028, 0.0705177)
# distSq: 0.10426608317727296
# Collision
# w: (-0.8863730000000001, -0.807266)
# u: (0.08789822736200421, 0.08005348810220492)

# using time_step: 0.033
# relative_velocity: (-0.118345, -0.102089)
# relative_position: (0.0768028, 0.0705177)
# distSq: 0.10426608317727296
# Collision
# w: (-2.4457025757575757, -2.2389889999999997)
# u: (0.031770384716383014, 0.02908511550457664)

# using time_step: 0.33
# relative_velocity: (-0.118345, -0.102089)
# relative_position: (0.0768028, 0.0705177)
# distSq: 0.10426608317727296
# Collision
# w: (-0.35108075757575763, -0.31577900000000003)
# u: (0.10775411890936654, 0.09691926196706604)
relative_velocity = (velocity[0] - other_velocity[0], velocity[1] - other_velocity[1])
relative_position = (other_pos[0] - pos[0], other_pos[1] - pos[1])
relative_velocity = (-0.118345,-0.102089)
relative_position = (0.0768028,0.0705177)
print("relative_velocity:", relative_velocity)
print("relative_position:", relative_position)

distSq = math.sqrt(relative_position[0] * relative_position[0] + relative_position[1] * relative_position[1])
print("distSq:", distSq)
if distSq > combined_radius:
    print("No collision")
else:
    print("Collision")
    w = (relative_velocity[0] - relative_position[0] * inv_time_step, relative_velocity[1] - relative_position[1] * inv_time_step)
    wLength = math.sqrt(w[0] * w[0] + w[1] * w[1])
    unit_w = (w[0] / wLength, w[1] / wLength)

    print("w:", w)
    u = ((combined_radius *  inv_time_step - wLength) * unit_w[0], (combined_radius *  inv_time_step - wLength) * unit_w[1])

    print("u:", u)