from typing import List

import numpy as np
from gymnasium import ActionWrapper, Env
from gymnasium.spaces import MultiDiscrete


# generate a template multidiscreteaction wrapper for me (don't modify the existing one)
class MultiDiscreteActionWrapper(ActionWrapper):
    """A Gymnasium `ActionWrapper` that translates craftium's `Dict` action space into a multi-discretized action space [`MultiDiscrete`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete).

    :param env: The environment to wrap.
    :param actions: A list of strings containing the names of the actions that will consititute the new action space.
    :params mouse_mov: Magnitude of the mouse movement. Must be in the [0, 1] range, else it will be clipped.
    """

    def __init__(self, env: Env):
        ActionWrapper.__init__(self, env)

        actions = [
            [
                # base
                "forward",
                "backward",
                "left",
                "right",
                "forward+left",
                "forward+right",
                "backward+left",
                "backward+right",
                # jump
                "jump",
                "forward+jump",
                "backward+jump",
                "left+jump",
                "right+jump",
                "forward+left+jump",
                "forward+right+jump",
                "backward+left+jump",
                "backward+right+jump",
                # sneak
                "sneak",
                "forward+sneak",
                "backward+sneak",
                "left+sneak",
                "right+sneak",
                "forward+left+sneak",
                "forward+right+sneak",
                "backward+left+sneak",
                "backward+right+sneak",
                # aux1 (sprint)
                "aux1",
                "forward+aux1",
                "backward+aux1",
                "left+aux1",
                "right+aux1",
                "forward+left+aux1",
                "forward+right+aux1",
                "backward+left+aux1",
                "backward+right+aux1",
                # jump + sneak
                "jump+sneak",
                "forward+jump+sneak",
                "backward+jump+sneak",
                "left+jump+sneak",
                "right+jump+sneak",
                "forward+left+jump+sneak",
                "forward+right+jump+sneak",
                "backward+left+jump+sneak",
                "backward+right+jump+sneak",
                # jump + aux1
                "jump+aux1",
                "forward+jump+aux1",
                "backward+jump+aux1",
                "left+jump+aux1",
                "right+jump+aux1",
                "forward+left+jump+aux1",
                "forward+right+jump+aux1",
                "backward+left+jump+aux1",
                "backward+right+jump+aux1",
                # sneak + aux1
                "sneak+aux1",
                "forward+sneak+aux1",
                "backward+sneak+aux1",
                "left+sneak+aux1",
                "right+sneak+aux1",
                "forward+left+sneak+aux1",
                "forward+right+sneak+aux1",
                "backward+left+sneak+aux1",
                "backward+right+sneak+aux1",
                # jump + sneak + aux1
                "jump+sneak+aux1",
                "forward+jump+sneak+aux1",
                "backward+jump+sneak+aux1",
                "left+jump+sneak+aux1",
                "right+jump+sneak+aux1",
                "forward+left+jump+sneak+aux1",
                "forward+right+jump+sneak+aux1",
                "backward+left+jump+sneak+aux1",
                "backward+right+jump+sneak+aux1",
            ],
            [
                "dig",
                "place+slot_1",
                "place+slot_2",
                "place+slot_3",
                "place+slot_4",
                "place+slot_5",
                "place+slot_6",
                "place+slot_7",
                "place+slot_8",
                "place+slot_9",
            ],
            # [
            #     "slot_1",
            #     "slot_2",
            #     "slot_3",
            #     "slot_4",
            #     "slot_5",
            #     "slot_6",
            #     "slot_7",
            #     "slot_8",
            #     "slot_9",
            #     "drop",
            # ],
            ["mouse x-0.1", "mouse x+0.1"],
            ["mouse y-0.1", "mouse y+0.1"],
        ]

        # base actions
        base_actions = {
            "forward": 0,
            "backward": 1,
            "left": 2,
            "right": 3,
            "jump": 4,
            "aux1": 5,
            "sneak": 6,
            "dig": 7,
            "place": 8,
            "drop": 9,
            "slot_1": 10,
            "slot_2": 11,
            "slot_3": 12,
            "slot_4": 13,
            "slot_5": 14,
            "slot_6": 15,
            "slot_7": 16,
            "slot_8": 17,
            "slot_9": 18,
            "mouse x-0.1": 19,
            "mouse x+0.1": 20,
            "mouse y-0.1": 21,
            "mouse y+0.1": 22,
        }
        multihot_actions = []
        noop_multihot_action = np.zeros(len(base_actions), dtype=bool)
        for action_group in actions:
            multihot_action_group = []
            for action in action_group:
                multihot_action = noop_multihot_action.copy()
                if "+" in action and not action.startswith("mouse"):
                    for ac in action.split("+"):
                        if ac in base_actions:
                            multihot_action[base_actions[ac]] = 1
                        else:
                            raise ValueError(f"Action {ac} not recognized.")

                else:
                    multihot_action[base_actions[action]] = 1
                multihot_action_group.append(multihot_action)
            multihot_actions.append(multihot_action_group)

        self.actions = actions
        self.base_actions = base_actions
        self.multihot_actions = multihot_actions
        self.action_space = MultiDiscrete([len(action_category) + 1 for action_category in actions])

    def process(self, action: List[int] | np.ndarray):
        assert len(action) == len(self.actions)
        assert all(0 <= act <= len(self.actions[i]) for i, act in enumerate(action))

        res = {}
        mouse = [0, 0]
        for i, act in enumerate(action):
            if act > 0:
                name = self.actions[i][act - 1]
                if name.startswith("mouse x"):
                    mouse[0] += float(name[7:])
                elif name.startswith("mouse y"):
                    mouse[1] += float(name[7:])
                elif "+" in name:
                    sub_actions = name.split("+")
                    for sub_act in sub_actions:
                        res[sub_act] = 1
                else:
                    res[name] = 1

        res["mouse"] = mouse
        return res

    def multihot(self, action: List[int] | np.ndarray):
        # Convert a multi-discrete action into a multihot vector containing base actions
        multihot_action = np.zeros(len(self.base_actions), dtype=bool)
        for i, act in enumerate(action):
            if act > 0:
                multihot_action = np.logical_or(multihot_action, self.multihot_actions[i][act - 1])
        return multihot_action

    def action(self, action):
        return self.process(action)
