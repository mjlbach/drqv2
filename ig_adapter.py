# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple
import igibson
from igibson.envs.igibson_env import iGibsonEnv

import enum
import os
import numpy as np
import gym
import gym.spaces
# from dm_env import specs

class StepType(enum.IntEnum):
  """Defines the status of a `TimeStep` within a sequence."""
  # Denotes the first `TimeStep` in a sequence.
  FIRST = 0
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = 1
  # Denotes the last `TimeStep` in a sequence.
  LAST = 2

  def first(self) -> bool:
    return self is StepType.FIRST

  def mid(self) -> bool:
    return self is StepType.MID

  def last(self) -> bool:
    return self is StepType.LAST

class TimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)

class TimeStepWrapper:
    def __init__(self, env):
        self._env = env

    def _convert_state(self, step_return, action):
        state, reward, done, info = step_return
        if done:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        return TimeStep(observation=state, reward=reward, discount=1, action=action, step_type=step_type)

    def step(self, action):
        step_return =  self._env.step(action)
        return self._convert_state(step_return, action)


    def observation_spec(self):
        return self._env.observation_space

    def action_spec(self):
        return self._env.action_space

    def reset(self):
        state = self._env.reset()
        action = np.zeros(self.action_spec().shape, dtype=self.action_spec().dtype)
        return TimeStep(observation=state, reward=0, discount=1, action=action, step_type=StepType.FIRST)

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionDTypeWrapper:
    def __init__(self, env, dtype=np.float32):
        self._env = env
        self._action_spec = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._env.action_space.shape,
            dtype=dtype
        )

    def step(self, action):
        action = action.astype(self.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper:
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        assert self._num_repeats > 0
        for _ in range(self._num_repeats):
            timestep = self._env.step(action)
            reward += (timestep.reward or 0.0) * timestep.discount
            #TODO(mjlbach) No discount for now
            discount *= 1.0
            if timestep.step_type == StepType.LAST:
                break

        return ExtendedTimeStep(reward=reward, discount=discount, step_type=timestep.step_type, observation=timestep.observation, action=action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper:
    def __init__(self, env, num_frames, pixels_key='rgb'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec.spaces.keys()

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        # self._obs_spec = specs.BoundedArray(shape=np.concatenate(
        #     [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
        #                                     dtype=np.uint8,
        #                                     minimum=0,
        #                                     maximum=255,
        #                                     name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)



class ExtendedTimeStepWrapper:
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, config=None):
    # make sure reward is not visualized
    config="/home/michael/Repositories/lab/drqv2/search.yml"
    import pdb; pdb.set_trace()
    env = iGibsonEnv(config_file=config, mode="gui_non_interactive")
    # add wrappers
    env = TimeStepWrapper(env)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    # TODO(mjlbach): Ensure actions are bounded between -1 and 1
    # TODO(mjlbach): Ensure observations are 84 x 84 pixels
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, 'rgb')
    env = ExtendedTimeStepWrapper(env)
    return env

if __name__ == "__main__":
    make('test', 3, 3, 0)
