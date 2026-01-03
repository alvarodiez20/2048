import os
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator

def get_latest_episode_score(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    if 'dqn_shaped/score' in ea.Tags()['scalars']:
        events = ea.Scalars('dqn_shaped/score')
        return events[-1].step if events else 0
    return 0

log_dir = 'rl/runs/dqn_shaped_20260102_025624'
try:
    latest = get_latest_episode_score(log_dir)
    print(f"LATEST_EPISODE: {latest}")
except Exception as e:
    print(f"ERROR: {e}")
