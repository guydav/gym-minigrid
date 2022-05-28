from gym_minigrid.minigrid import *
from gym_minigrid.roomgrid import MiniGridEnv
from gym_minigrid.register import register


COLOR_TASK = 'color'
MAGNITUDE_TASK = 'magnitude'
PARITY_TASK = 'parity'

NUMBER_TASKS = (
    COLOR_TASK, MAGNITUDE_TASK, PARITY_TASK
)

DEFAULT_COLOR_INDICES = [1, 2]


class NumberTasksEnv(MiniGridEnv):
    def __init__(self, size=9, shuffle_task_locations=True, visualize_task=False,
                 task=None, color_indices=DEFAULT_COLOR_INDICES, seed=None):

        self.size = size
        self.grid_mid = (self.size + 1) // 2
        self.possible_task_locations = [
            (1, 1), (1, self.grid_mid), (1, self.size),
            (self.grid_mid, 1), (self.grid_mid, self.size),
            (self.size, 1), (self.size, self.grid_mid), (self.size, self.size)
        ]
        self.task_locations = set()
        self.digit = None
        self.correct_color_index = None
        self.color_indices = color_indices

        self.mission = 'Figure out which number task is active and perform it'
    
        super().__init__(grid_size=size + 2, max_steps=4*size*size, see_through_walls=True, seed=seed)

        self.shuffle_task_locations = shuffle_task_locations
        self.visualize_task = visualize_task

        if task is None:
            task = self.np_random.choice(NUMBER_TASKS)

        self.task = task
        

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.digit = self.np_random.integers(0, 10)
        self.correct_color_index = self.np_random.integers(0, 2)
        stimulus = Text(str(self.digit), 
            color=IDX_TO_COLOR[self.color_indices[self.correct_color_index]], 
            can_pickup=False)
        self.put_obj(stimulus, self.grid_mid, self.grid_mid)
        self.agent_pos = (self.grid_mid, self.grid_mid + 1)
        self.agent_dir = self.np_random.randint(0, len(DIR_TO_VEC))

        # place tasks
        if len(self.task_locations) == 0 or self.shuffle_task_locations:
            self._place_tasks()

        # TODO: do something with the visualize_task flag

    def _place_tasks(self):
        self.task_locations = set()
        location_permutation = self.np_random.permutation(len(self.possible_task_locations))

        for task_index, task in enumerate(NUMBER_TASKS):
            if task == COLOR_TASK:
                stimuli = [Ball(color=IDX_TO_COLOR[self.color_indices[self.correct_color_index]]),
                    Ball(color=IDX_TO_COLOR[self.color_indices[(self.correct_color_index + 1) % 2]])]
            
            elif task == MAGNITUDE_TASK:
                stimuli = [Text('+', color='grey'), Text('-', color='grey')]

            elif task == PARITY_TASK:
                stimuli = [Text('0', color='grey'), Text('1', color='grey')]

            for stim_index, stim in enumerate(stimuli):
                location = self.possible_task_locations[location_permutation[task_index * 2 + stim_index]]
                self.task_locations.add(location)
                self.put_obj(stim, *location)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            print(f'Task: {self.task}')
            task_stim = self.carrying

            if task_stim is None:
                # TODO: is this 0 reward, or -1?
                return obs, 0, done, info

            elif self.task == COLOR_TASK:
                if isinstance(task_stim, Ball):
                    reward = 1 if task_stim.color == IDX_TO_COLOR[self.correct_color_index] else -1
                else:
                    reward = -1

            elif self.task == MAGNITUDE_TASK:
                if isinstance(task_stim, Text) and task_stim.text in ('+', '-'):
                    reward = 1 if (task_stim.text == '+') == (self.digit >= 5) else -1
                else:
                    reward = -1
                
            elif self.task == PARITY_TASK:
                if isinstance(task_stim, Text) and task_stim.text in ('0', '1'):
                    reward = 1 if (task_stim.text == '0') == (self.digit % 2 == 0) else -1
                else:
                    reward = -1

            else:
                raise ValueError(f'Unknown task: {self.task}')

            done = True

        return obs, reward, done, info


class NumberTask9x9(NumberTasksEnv):
    def __init__(self, task=None, color_indices=DEFAULT_COLOR_INDICES, seed=None):
        super().__init__(size=9, shuffle_task_locations=True, 
            visualize_task=False, task=task, 
            color_indices=color_indices, seed=seed)

register(
    id='MiniGrid-NumberTask9x9-v0',
    entry_point='gym_minigrid.envs:NumberTask9x9'
)
