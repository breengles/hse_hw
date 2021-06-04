game_1v1 = {
    "game": {
        "num_obsts": 16,
        "num_preds": 1,
        "num_preys": 1,
        "x_limit": 12,
        "y_limit": 12,
        "obstacle_radius_bounds": [0.8, 2.0],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/100,
        "frameskip": 5
    },
    "environment": {
        "frameskip": 5,
        "time_limit": 1000
    }
}

simple = {
    "game": {
        "num_obsts": 0,
        "num_preds": 1,
        "num_preys": 1,
        "x_limit": 6,
        "y_limit": 6,
        "obstacle_radius_bounds": [0.8, 2.0],
        "prey_radius": 0.8,
        "predator_radius": 1.0,
        "predator_speed": 6.0,
        "prey_speed": 9.0,
        "world_timestep": 1/100,
        "frameskip": 5
    },
    "environment": {
        "frameskip": 5,
        "time_limit": 1000
    }
}

