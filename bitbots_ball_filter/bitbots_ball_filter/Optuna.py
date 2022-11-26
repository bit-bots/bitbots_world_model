import optuna
import subprocess
import os


def objective(trial):
    x = trial.suggest_int("x", 1, 10)
    print(x)

    os.system("ros2 param set /bitbots_ball_filter filter_reset_distance " + str(x))
    print("ros2 param set /bitbots_ball_filter filter_reset_distance " + str(x))
    return os.system("ros2 param get /bitbots_ball_filter filter_reset_distance")


study = optuna.create_study()
study.optimize(objective, n_trials=3)

best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x-2)^2: {}".format(found_x, (found_x - 2) ** 2))