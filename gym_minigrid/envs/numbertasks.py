from collections import defaultdict
import typing

from gym_minigrid.minigrid import *
from gym_minigrid.roomgrid import MiniGridEnv
from gym_minigrid.register import register
import enum


class NumberTaskType(enum.Enum):
    color = 'color'
    magnitude = 'magnitude'
    parity = 'parity'


DEFAULT_COLOR_INDICES = [1, 2]


class NumberTasksGridEnv(MiniGridEnv):
    def __init__(self, size: int = 9, shuffle_task_locations: bool = True, visualize_task: bool = False,
                task: typing.Optional[NumberTaskType] = None, 
                color_indices: typing.Sequence[int] = DEFAULT_COLOR_INDICES, 
                seed: typing.Optional[int] = None,
                positive_reward: typing.Optional[int] = None,
                negative_reward: int = 0, 
        ):

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
            self.seed(seed)
            task = self.np_random.choice(list(NumberTaskType))
        elif isinstance(task, str):
            task = NumberTaskType(task)

        self.task = task
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.digit = self.np_random.randint(0, 10)
        self.correct_color_index = self.np_random.randint(0, 2)
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

        for task_index, task in enumerate(NumberTaskType):
            if task == NumberTaskType.color:
                stimuli = [Ball(color=IDX_TO_COLOR[self.color_indices[self.correct_color_index]]),
                    Ball(color=IDX_TO_COLOR[self.color_indices[(self.correct_color_index + 1) % 2]])]
            
            elif task == NumberTaskType.magnitude:
                stimuli = [Text('+', color='grey'), Text('-', color='grey')]

            elif task == NumberTaskType.parity:
                stimuli = [Text('0', color='grey'), Text('1', color='grey')]

            for stim_index, stim in enumerate(stimuli):
                location = self.possible_task_locations[location_permutation[task_index * 2 + stim_index]]
                self.task_locations.add(location)
                self.put_obj(stim, *location)

    def step(self, action: MiniGridEnv.Actions):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            print(f'Task: {self.task}')
            task_stim = self.carrying

            if task_stim is None:
                # TODO: is this 0 reward, or -1?
                return obs, 0, done, info

            elif self.task == NumberTaskType.color:
                if isinstance(task_stim, Ball):
                    reward = 1 if task_stim.color == IDX_TO_COLOR[self.color_indices[self.correct_color_index]]  else -1
                else:
                    reward = -1

            elif self.task == NumberTaskType.magnitude:
                if isinstance(task_stim, Text) and task_stim.text in ('+', '-'):
                    reward = 1 if (task_stim.text == '+') == (self.digit >= 5) else -1
                else:
                    reward = -1
                
            elif self.task == NumberTaskType.parity:
                if isinstance(task_stim, Text) and task_stim.text in ('0', '1'):
                    reward = 1 if (task_stim.text == '0') == (self.digit % 2 == 0) else -1
                else:
                    reward = -1

            else:
                raise ValueError(f'Unknown task: {self.task}')

            done = True

        if reward == 1:
            reward = self._reward() if self.positive_reward is None else self.positive_reward
        elif reward == -1:
            reward = self.negative_reward

        return obs, reward, done, info



class NumberTasksTMaze(MiniGridEnv):
    def __init__(self, size: typing.Union[int, typing.Tuple[int, int]] = 5, 
                 visualize_task: bool = False, 
                 task: typing.Optional[NumberTaskType] = None, switch_tasks: bool = False,
                 show_all_tasks: bool = False,  shuffle_task_locations: bool = True,
                 color_indices: typing.Sequence[int] = DEFAULT_COLOR_INDICES, 
                 seed: typing.Optional[int] = None,
                 positive_reward: typing.Optional[float] = None,
                 negative_reward: float = 0,
                 step_reward: float = 0,
                 min_agent_view_size: int = 101,):

        if isinstance(size, int):
            width = height = size
        else:
            width, height = size
        
        self.w_mid = (width + 1) // 2
        self.h_mid = (height + 1) // 2

        self.goal_locations = [(1, 1), (width, 1)]
        self.task_locations = defaultdict(list)
        self.locations_to_task_markers = {}
        self.digit = None
        self.correct_color_index = None
        self.color_indices = color_indices

        self.shuffle_task_locations = shuffle_task_locations
        self.visualize_task = visualize_task
        self.switch_tasks = switch_tasks
        self.show_all_tasks = show_all_tasks

        self.mission = 'Figure out which number task is active and perform it'
        
        if task is None:
            self.seed(seed)
            task = self.np_random.choice(list(NumberTaskType))
        elif isinstance(task, str):
            task = NumberTaskType(task)

        self.task = task
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.step_reward = step_reward
    
        super().__init__(width=width + 2, height=height + 2,
            max_steps=4*(max(width, height) ** 2), 
            see_through_walls=True, seed=seed, agent_view_size=min(width + 2, height + 2, min_agent_view_size))

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if height - 1 > 2:
            for i in range(1, width - 1):
                if i != self.w_mid:
                    self.grid.vert_wall(i, 2, height - 3)

        self.digit = self.np_random.randint(0, 10)
        self.correct_color_index = self.np_random.randint(0, 2)
        stimulus = Text(str(self.digit), 
            color=IDX_TO_COLOR[self.color_indices[self.correct_color_index]], 
            can_pickup=False)
        self.put_obj(stimulus, self.w_mid, 0)
        self.agent_pos = (self.w_mid, height - 2)
        self.agent_dir = 3

        for goal_loc in self.goal_locations:
            self.put_obj(Goal(), *goal_loc)

        # place tasks
        self._place_tasks(width, height)

        # TODO: do something with the visualize_task flag

    def _place_tasks(self, width, height):
        if len(self.task_locations) == 0 or self.shuffle_task_locations:
            self.task_locations = defaultdict(list)

            if self.show_all_tasks:
                all_locations = [
                    ((0, 1), (width - 1, 1)),
                    ((1, 0), (width - 2, 0)),
                    ((1, 2), (width - 2, 2)),
                ]
                if self.shuffle_task_locations:
                    all_locations_permutation = self.np_random.permutation(len(all_locations))
                else:
                    all_locations_permutation = np.arange(len(all_locations))

                for task_index, task in enumerate(NumberTaskType):
                    locations = all_locations[all_locations_permutation[task_index]]
                    if self.shuffle_task_locations:
                        permutation = self.np_random.permutation(len(locations))
                    else:
                        permutation = np.arange(len(locations))

                    if task == NumberTaskType.color:
                        stimuli = [Ball(color=IDX_TO_COLOR[self.color_indices[self.correct_color_index]]),
                            Ball(color=IDX_TO_COLOR[self.color_indices[(self.correct_color_index + 1) % 2]])]
                    
                    elif task == NumberTaskType.magnitude:
                        stimuli = [Text('+', color='grey'), Text('-', color='grey')]

                    elif task == NumberTaskType.parity:
                        stimuli = [Text('0', color='grey'), Text('1', color='grey')]

                    for loc_index, loc in enumerate(locations):
                        stimulus = stimuli[permutation[loc_index]]
                        self.task_locations[loc_index].append(stimulus)
                        self.locations_to_task_markers[loc] = stimulus
                        self.put_obj(stimulus, *loc)
            else:
                locations = [(0, 1), (width - 1, 1)]
                if self.shuffle_task_locations:
                    permutation = self.np_random.permutation(len(locations))
                else:
                    permutation = np.arange(len(locations))

                if self.task == NumberTaskType.color:
                    stimuli = [Ball(color=IDX_TO_COLOR[self.color_indices[self.correct_color_index]]),
                        Ball(color=IDX_TO_COLOR[self.color_indices[(self.correct_color_index + 1) % 2]])]
                
                elif self.task == NumberTaskType.magnitude:
                    stimuli = [Text('+', color='grey'), Text('-', color='grey')]

                elif self.task == NumberTaskType.parity:
                    stimuli = [Text('0', color='grey'), Text('1', color='grey')]

                for loc_index, loc in enumerate(locations):
                    stimulus = stimuli[permutation[loc_index]]
                    self.task_locations[loc_index].append(stimulus)
                    self.locations_to_task_markers[loc] = stimulus
                    self.put_obj(stimulus, *loc)

        else:
            for loc, stimlus in self.locations_to_task_markers.items():
                self.put_obj(stimlus, *loc)

    
    def step(self, action: MiniGridEnv.Actions):
        obs, reward, done, info = super().step(action)

        agent_pos = tuple(self.agent_pos)
        if agent_pos in self.goal_locations:
            agent_pos_index = self.goal_locations.index(agent_pos)

            if self.task == NumberTaskType.color:
                task_stim = list(filter(lambda obj: isinstance(obj, Ball), self.task_locations[agent_pos_index]))[0]
                reward = 1 if task_stim.color == IDX_TO_COLOR[self.color_indices[self.correct_color_index]] else -1

            elif self.task == NumberTaskType.magnitude:
                task_stim = list(filter(lambda obj: isinstance(obj, Text) and obj.text in ('+', '-'), self.task_locations[agent_pos_index]))[0]
                reward = 1 if (task_stim.text == '+') == (self.digit >= 5) else -1
                
            elif self.task == NumberTaskType.parity:
                task_stim = list(filter(lambda obj: isinstance(obj, Text) and obj.text in ('0', '1'), self.task_locations[agent_pos_index]))[0]
                reward = 1 if (task_stim.text == '0') == (self.digit % 2 == 0) else -1

            else:
                raise ValueError(f'Unknown task: {self.task}')

            done = True

        if reward == 1:
            reward = self._reward() if self.positive_reward is None else self.positive_reward
        elif reward == -1:
            reward = self.negative_reward
        else:
            reward = self.step_reward

        return obs, reward, done, info



class NumberTaskGrid9x9(NumberTasksGridEnv):
    def __init__(self, task=None, color_indices=DEFAULT_COLOR_INDICES, seed=None, **kwargs):
        super().__init__(size=9, shuffle_task_locations=True, 
            visualize_task=False, task=task, 
            color_indices=color_indices, seed=seed, **kwargs)


class NumberTasksTMaze5(NumberTasksTMaze):
    def __init__(self, visualize_task: bool = False, switch_tasks: bool = False,
                 show_all_tasks: bool = False, shuffle_task_locations: bool = False,
                 task: typing.Optional[NumberTaskType] = None, 
                 color_indices: typing.Sequence[int] = DEFAULT_COLOR_INDICES,
                 seed: typing.Optional[int] = None, **kwargs):
        super().__init__(size=5,
            visualize_task=visualize_task, task=task, switch_tasks=switch_tasks,
            show_all_tasks=show_all_tasks, shuffle_task_locations=shuffle_task_locations,
            color_indices=color_indices, seed=seed, **kwargs)


class NumberTasksNosePoke(NumberTasksTMaze):
    def __init__(self, visualize_task: bool = False, switch_tasks: bool = False,
                 show_all_tasks: bool = False, shuffle_task_locations: bool = False,
                 task: typing.Optional[NumberTaskType] = None, 
                 color_indices: typing.Sequence[int] = DEFAULT_COLOR_INDICES,
                 seed: typing.Optional[int] = None, **kwargs):
        super().__init__(size=(3, 1),
            visualize_task=visualize_task, task=task, switch_tasks=switch_tasks,
            show_all_tasks=show_all_tasks, shuffle_task_locations=shuffle_task_locations,
            color_indices=color_indices, seed=seed, **kwargs)


register(
    id='MiniGrid-NumberTaskGrid9x9-v0',
    entry_point='gym_minigrid.envs:NumberTaskGrid9x9'
)

register(
    id='MiniGrid-NumberTasksTMaze5-v0',
    entry_point='gym_minigrid.envs:NumberTasksTMaze5'
)

register(
    id='MiniGrid-NumberTasksNosePoke-v0',
    entry_point='gym_minigrid.envs:NumberTasksNosePoke'
)

