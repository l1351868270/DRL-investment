import gym
from gym import spaces
import pygame
import numpy as np
import pandas as pd

from drl_investment.data.tdx import DataRaw


class TDXRawEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self, render_mode=None, data: np.array = None, columns: list[str] = None):
        if data is None:
            raise Exception(f'data must not be None')
        if columns is None:
            raise Exception(f'index must not be None')
        
        self._min_len = 100
        self._len = data.shape[0]
        if self._len < self._min_len:
            raise Exception(f'data length must large than {self._min_len}')
        
        self._index = 0
        self._data = data
        self._columns = columns

        
        self.window_size = 512  # The size of the PyGame window
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(
        #     {
        #         'agent': spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )

        self.observation_space = spaces.Box(0, np.inf, shape=(1, ), dtype=np.float32)
        # We have 3 actions, corresponding to 'buy', 'skip', 'sell'
        self.action_space = spaces.Discrete(3)

        # '''
        # The following dictionary maps abstract actions from `self.action_space` to 
        # the direction we will walk in if that action is taken.
        # I.e. 0 corresponds to 'right', 1 to 'up' etc.
        # '''
        # self._action_to_direction = {
        #     0: np.array([1, 0]),
        #     1: np.array([0, 1]),
        #     2: np.array([-1, 0]),
        #     3: np.array([0, -1]),
        # }

        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        '''
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        '''
        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._data[self._index]

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._index = self.np_random.integers(0, int(self._len*0.625), size=1, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, action):
        terminated = self._index >= self._len
        reward = 0 if self._index==0 else self._data[self._index]/self._data[self._index-1]-1  # open/ref(open,1) - 1
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Finally, add some gridlines
        for i, name in enumerate(self._columns):
            pygame.draw.lines(
                canvas,
                0,
                False,
                enumerate(self.data[:self._index+1, i].to_list()),
                width=3,
            )

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata['render_fps'])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()