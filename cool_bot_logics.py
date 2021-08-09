import numpy as np
from numpy.linalg import norm
from rlgym.utils.gamestates import GameState
import math

THROTTLE = 0
STEER = 1
PITCH = 2
YAW = 3
ROLL = 4
JUMP = 5
BOOST = 6
HANDBRAKE = 7


def supersonicml_atba(obs: np.ndarray, state: GameState, p_id: int) -> np.ndarray:
    out = np.zeros(8, dtype=float)

    ball = state.ball
    player = state.players[p_id]
    car = player.car_data
    team_sign = player.team_num * 2 - 1

    gravity = np.array([0, 0, -650], dtype=float)

    T = min(0.5, norm(car.position[:2] - ball.position[:2]) / max(float(1400), 50 + norm(car.linear_velocity)))
    future_ball_pos = ball.position + ball.linear_velocity * T + 0.5 * gravity * T * T
    future_ball_pos[2] = np.clip(future_ball_pos[2], 95, 2000)

    if future_ball_pos[2] > 100 and norm(car.position[:2] - future_ball_pos[:2]) > 500:
        future_ball_pos[2] = 0  # target in air
    else:
        future_ball_pos[2] = max(float(17), future_ball_pos[2] - 70)

    if team_sign * car.position[1] < team_sign * future_ball_pos[1]:
        target = np.array([0, team_sign * 5120, 17], dtype=float)
    else:
        target = future_ball_pos

    target_local = np.array([np.dot(target - car.position, car.forward()), np.dot(target - car.position, car.right())])
    angle = math.atan2(target_local[1], target_local[0])

    out[STEER] = np.clip(3 * angle, -1, 1)
    out[THROTTLE] = np.clip(norm(car.position[:2] - target[:2]) / 100, 0.05, 1)

    if abs(angle) > 1 and np.dot(car.linear_velocity, car.forward()) > 800:
        out[THROTTLE] = -1

    if target_local[0] > 1000 and abs(angle) < 0.1 and np.dot(car.linear_velocity, car.forward()) < 1800:
        out[BOOST] = 1
    elif abs(angle) > 0.9 and abs(np.dot(car.angular_velocity, car.up())) < 3 and np.dot(car.linear_velocity,
                                                                                         car.forward()) > 300:
        out[HANDBRAKE] = 1

    return out
