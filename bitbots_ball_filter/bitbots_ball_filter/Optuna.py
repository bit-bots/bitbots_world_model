#! /usr/bin/env python3
import optuna
from subprocess import call
print(call(["source ~/colcon_ws/install/setup.bash"]))
print(call(["ros2 node list"]))

def objective(trial):
    x = trial.suggest_float("x", 1, 10)
    print(x)

    call(["ros2 param set /bitbots_ball_filter filter_reset_distance", str(x)])
    return x


study = optuna.create_study()
study.optimize(objective, n_trials=100)

best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x-2)^2: {}".format(found_x, (found_x - 2) ** 2))