import time
from typing import Dict, Text, Tuple
from gym import register
import numpy as np
import math
import random
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane, SineLane
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.road.regulation import RegulatedRoad

Observation = np.ndarray


class SquareEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                # "absolute": True,
                "features": ["presence", "x", "y", "vx", "vy"],  #
                "features_range": {"x": [-250, 250], "y": [-250, 250], "vx": [-30, 30], "vy": [-30, 30]},
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "Create_vehicles": True,
            "screen_width": 960,
            "screen_height":540,
            "incoming_vehicle_destination": None,
            "spawn_probability": 0.6,
            "lanes_count": 2,
            "vehicles_count": 60,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 80,  # [s]40
            "ego_spacing": 0.15,
            "vehicles_density": 0.8,
            # "b_reward": 10,
            # "d_reward": 10,
            # "f_reward": 10,
            # "h_reward": 10,
            "collision_reward": -80,    # The reward received when colliding with a vehicle.    发生碰撞的奖励
            # "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 3,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "low_speed_reward": -3,
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "low_reward_speed_range": [10, 20],
            "normalize_reward": False,
            "offroad_terminal": True,
            # "real_time_rendering": True,
            "is_truncated": True,
            "check_safe" : True
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        # self.t = 0
        # self.cls = 0
        # self.Arrival = [1,1,1,1]

    def _create_road(self) -> None:
        net = RoadNetwork()

        r1 = 5

        # 直道ab
        net.add_lane("a", "b",
                     StraightLane([-200, 205], [200, 205], line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  width=5, speed_limit=30))
        net.add_lane("a", "b",
                     StraightLane([-200, 210], [200, 210], line_types=(LineType.NONE, LineType.STRIPED),
                                  width=5, speed_limit=30))
        net.add_lane("a", "b",
                     StraightLane([-200, 215], [200, 215], line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  width=5, speed_limit=30))

        # 弯道bc
        center1 = [200, 200]
        net.add_lane("b", "c",
                     CircularLane(center1, r1, np.deg2rad(90), np.deg2rad(0), width=5,
                                  clockwise=False, line_types=[LineType.CONTINUOUS, LineType.NONE],
                                  speed_limit=30))
        net.add_lane("b", "c",
                     CircularLane(center1, r1+5, np.deg2rad(90), np.deg2rad(0), width=5,
                                  clockwise=False, line_types=[LineType.STRIPED, LineType.NONE],
                                  speed_limit=30))
        net.add_lane("b", "c",
                     CircularLane(center1, r1 + 10, np.deg2rad(90), np.deg2rad(0), width=5,
                                  clockwise=False, line_types=[LineType.STRIPED, LineType.CONTINUOUS],
                                  speed_limit=30))
        # 直道cd
        net.add_lane("c", "d",
                     StraightLane([205, 200], [205, -200], line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  width=5, speed_limit=30))
        net.add_lane("c", "d",
                     StraightLane([210, 200], [210, -200], line_types=(LineType.NONE, LineType.STRIPED),
                                  width=5, speed_limit=30))
        net.add_lane("c", "d",
                     StraightLane([215, 200], [215, -200], line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  width=5, speed_limit=30))
        # 弯道de
        center2 = [200, -200]
        net.add_lane("d", "e",
                     CircularLane(center2, r1, np.deg2rad(0), np.deg2rad(-90), width=5,
                                  clockwise=False, line_types=[LineType.CONTINUOUS, LineType.NONE],
                                  speed_limit=30))
        net.add_lane("d", "e",
                     CircularLane(center2, r1 + 5, np.deg2rad(0), np.deg2rad(-90), width=5,
                                  clockwise=False, line_types=[LineType.STRIPED, LineType.NONE],
                                  speed_limit=30))
        net.add_lane("d", "e",
                     CircularLane(center2, r1 + 10, np.deg2rad(0), np.deg2rad(-90), width=5,
                                  clockwise=False, line_types=[LineType.STRIPED, LineType.CONTINUOUS],
                                  speed_limit=30))
        # 直道ef
        net.add_lane("e", "f",
                     StraightLane([200, -205], [-200, -205], line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  width=5, speed_limit=30))
        net.add_lane("e", "f",
                     StraightLane([200, -210], [-200, -210], line_types=(LineType.NONE, LineType.STRIPED),
                                  width=5, speed_limit=30))
        net.add_lane("e", "f",
                     StraightLane([200, -215], [-200, -215], line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  width=5, speed_limit=30))
        # 弯道fg
        center3 = [-200, -200]
        net.add_lane("f", "g",
                     CircularLane(center3, r1, np.deg2rad(-90), np.deg2rad(-180), width=5,
                                  clockwise=False, line_types=[LineType.CONTINUOUS, LineType.NONE],
                                  speed_limit=30))
        net.add_lane("f", "g",
                     CircularLane(center3, r1 + 5, np.deg2rad(-90), np.deg2rad(-180), width=5,
                                  clockwise=False, line_types=[LineType.STRIPED, LineType.NONE],
                                  speed_limit=30))
        net.add_lane("f", "g",
                     CircularLane(center3, r1 + 10, np.deg2rad(-90), np.deg2rad(-180), width=5,
                                  clockwise=False, line_types=[LineType.STRIPED, LineType.CONTINUOUS],
                                  speed_limit=30))
        # 直道gh
        net.add_lane("g", "h",
                     StraightLane([-205, -200], [-205, 200], line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  width=5, speed_limit=30))
        net.add_lane("g", "h",
                     StraightLane([-210, -200], [-210, 200], line_types=(LineType.STRIPED, LineType.NONE),
                                  width=5, speed_limit=30))
        net.add_lane("g", "h",
                     StraightLane([-215, -200], [-215, 200], line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  width=5, speed_limit=30))

        # 弯道ha
        center3 = [-200, 200]
        net.add_lane("h", "a",
                     CircularLane(center3, r1, np.deg2rad(180), np.deg2rad(90), width=5,
                                  clockwise=False, line_types=[LineType.CONTINUOUS, LineType.NONE],
                                  speed_limit=30))
        net.add_lane("h", "a",
                     CircularLane(center3, r1 + 5, np.deg2rad(180), np.deg2rad(90), width=5,
                                  clockwise=False, line_types=[LineType.STRIPED, LineType.NONE],
                                  speed_limit=30))
        net.add_lane("h", "a",
                     CircularLane(center3, r1 + 10, np.deg2rad(180), np.deg2rad(90), width=5,
                                  clockwise=False, line_types=[LineType.STRIPED, LineType.CONTINUOUS],
                                  speed_limit=30))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road


    def _create_vehicles(self) -> None:
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        ego_lane = self.road.network.get_lane(("a", "b", 0))
        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, ego_lane.position(0, 0), heading=0, speed = 25)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.lane_create_random(self.road, lane_from = ["a", "c", "e", "g"], speed = 20, spacing = 1/self.config['vehicles_density'])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)


    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        # reward = reward - (self.time - self.t) * 0.8
        # self.t = self.time
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        # print(self.vehicle.speed)
        forward_speed = self.vehicle.speed     # * np.cos(self.vehicle.heading)
        # self.cls+=1
        # if self.cls%30 == 0:
        #     print(forward_speed)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        low_speed = utils.lmap(forward_speed, self.config["low_reward_speed_range"], [1, 0])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "low_speed_reward": np.clip(low_speed, 0, 1),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def safety_check(self, action: int):
        dt = 2 / self.config["policy_frequency"]
        self.error = [4, 4]
        self.available_actions_list = self.get_available_actions()
        # print(self.available_actions_list)
        _obs = self.observation_type.observe()
        _obs = np.delete(_obs, 0, axis=1)
        obs = self.obs_restore(_obs)
        print(obs)
        heading = self.controlled_vehicles[0].to_dict()['heading']
        if action in self.available_actions_list:
            if not self._is_safe(action,obs,heading,dt):
                print("有危险")
                return self.get_new_action(action,obs,heading,dt)
            else:
                return action
        else:
            return self.get_new_action(action,obs,heading,dt)

    def get_new_action(self,action,obs,heading,dt):
        action_list = []
        for i in self.available_actions_list:
            if i == action:
                continue
            if self._is_safe(i, obs, heading, dt):
                action_list.append(i)
        if len(action_list) == 0:
            print("没有安全动作")
            action = 4
            return action
        else:
            print("可用安全动作有：",action_list)
            action = random.choice(action_list)
            print("新选择动作为：", action)
            return action

    def _is_safe(self, action, obs, heading,dt):
        ego_vehicle = obs[0]
        npc_action = 1
        # IDLE
        if action == 1:
            vehicle = self.road.vehicles[0]
            for other in self.road.vehicles[1:]:
                if vehicle.new_handle_collisions(other, dt):
                        return False
            return True

        # FASTER
        elif action == 3:
            new_state = []
            vehicle = self.road.vehicles[0]
            # for other in self.road.vehicles[1:]:
            #     if vehicle.new_handle_collisions(other, dt):
            #         return False
            new_ego_vehicle = self.move(action, ego_vehicle, dt)
            for i in range(1,5):
                new_state.append(self.move(npc_action, obs[i], dt))
            for i in range(1,5):
                if self.error[0] > abs(new_ego_vehicle[0] - new_state[i-1][0]) and \
                    self.error[1] > abs(new_ego_vehicle[1] - new_state[i-1][1]):
                    return False
            return True

        # SLOWER
        elif action == 4:
            new_state = []
            new_ego_vehicle = self.move(action, ego_vehicle, dt)
            for i in range(1,5):
                new_state.append(self.move(npc_action, obs[i], dt))
            for i in range(1,5):
                if self.error[0] >abs(new_ego_vehicle[0] - new_state[i-1][0]) and \
                    self.error[1] >abs(new_ego_vehicle[1] - new_state[i-1][1]):
                    return False
            return True

        # LANE_RIGHT
        elif action == 2:
            offset = self.vehicle.speed * dt * 0.5
            A = (ego_vehicle[0] +5 * math.cos(heading), ego_vehicle[1] + 5 * math.sin(heading))
            B = (A[0] - 7.5 * math.cos(math.pi/2 - heading), A[1] + 7.5 * math.sin(math.pi/2 - heading))
            C = (ego_vehicle[0] - 4 * math.cos(heading), ego_vehicle[1] - 4 * math.sin(heading))
            D = (C[0] - 7.5 * math.cos(math.pi / 2 - heading), C[1] + 7.5 * math.sin(math.pi / 2 - heading))
            E = (B[0] + offset * math.cos(heading), B[1] + offset * math.sin(heading))
            F = (E[0] + 5 * math.cos(math.pi / 2 - heading), E[1] - 5 * math.sin(math.pi / 2 - heading))
            G = (B[0] + 5 * math.cos(math.pi / 2 - heading), B[1] - 5 * math.sin(math.pi / 2 - heading))
            polygon_1 = [A, B, C, D]
            polygon_2 = [G, B, E, F]
            for n in range(1, 5):
                p = (obs[n][0], obs[n][1])
                count_1 = 0
                count_2 = 0
                for i in range(4):
                    p1, p2 = polygon_1[i], polygon_1[(i + 1) % n]
                    if p1[1] == p2[1]:
                        continue
                    if p[1] < min(p1[1], p2[1]) or p[1] >= max(p1[1], p2[1]):
                        continue
                    x = (p[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    if x > p[0]:
                        count_1 += 1
                if count_1 % 2 == 1:
                    return False
                for i in range(4):
                    p1, p2 = polygon_2[i], polygon_2[(i + 1) % n]
                    if p1[1] == p2[1]:
                        continue
                    if p[1] < min(p1[1], p2[1]) or p[1] >= max(p1[1], p2[1]):
                        continue
                    x = (p[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    if x > p[0]:
                        count_2 += 1
                if count_2 % 2 == 1:
                    return False

            return True

        # LANE_LEFT
        elif action == 0:
            offset = self.vehicle.speed * dt * 0.5
            # print(f"offset为{offset},dt为{dt}")
            A = (ego_vehicle[0] + 5 * math.cos(heading), ego_vehicle[1] + 5 * math.sin(heading))
            B = (A[0] + 7.5 * math.cos(math.pi/2 - heading), A[1] - 7.5 * math.sin(math.pi/2 - heading))
            C = (ego_vehicle[0] - 4 * math.cos(heading), ego_vehicle[1] - 4 * math.sin(heading))
            D = (C[0] + 7.5 * math.cos(math.pi / 2 - heading), C[1] - 7.5 * math.sin(math.pi / 2 - heading))
            E = (B[0] + offset * math.cos(heading), B[1] + offset * math.sin(heading))
            F = (E[0] - 5 * math.cos(math.pi / 2 - heading), E[1] + 5 * math.sin(math.pi / 2 - heading))
            G = (B[0] - 5 * math.cos(math.pi / 2 - heading), B[1] + 5 * math.sin(math.pi / 2 - heading))
            polygon_1 = [A, B, C, D]
            polygon_2 = [G, B, E, F]
            for n in range(1, 5):
                p = (obs[n][0], obs[n][1])
                count_1 = 0
                count_2 = 0
                for i in range(4):
                    p1, p2 = polygon_1[i], polygon_1[(i + 1) % n]
                    if p1[1] == p2[1]:
                        continue
                    if p[1] < min(p1[1], p2[1]) or p[1] >= max(p1[1], p2[1]):
                        continue
                    x = (p[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    if x > p[0]:
                        count_1 += 1
                if count_1 % 2 == 1:
                    return False
                for i in range(4):
                    p1, p2 = polygon_2[i], polygon_2[(i + 1) % n]
                    if p1[1] == p2[1]:
                        continue
                    if p[1] < min(p1[1], p2[1]) or p[1] >= max(p1[1], p2[1]):
                        continue
                    x = (p[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    if x > p[0]:
                        count_2 += 1
                if count_2 % 2 == 1:
                    return False
            return True

    def move(self,action, state, dt):
        dt *= 4
        a = 1
        if action == 1:   # IDLE
            state[0] = state[0] + state[2] * dt
            state[1] = state[1] + state[3] * dt

        elif action == 3:   # FASTER
            state[0] = state[0] + 0.5 * a * dt**2 + state[2] * dt
            state[1] = state[1] + 0.5 * a * dt ** 2 + state[3] * dt
            state[2] = state[2] + a * dt # / math.sqrt(2)
            state[3] = state[3] + a * dt # / math.sqrt(2)
        elif action == 4:   # SLOWER
            state[0] = state[0] - 0.5 * a * dt**2 + state[2] * dt
            state[1] = state[1] - 0.5 * a * dt ** 2 + state[3] * dt
            state[2] = state[2] - a * dt # / math.sqrt(2)
            state[3] = state[3] - a * dt # / math.sqrt(2)

        return state

    def obs_restore(self,obs):
        for i in range(5):
            obs[i][0] = obs[i][0] * 250
            obs[i][1] = obs[i][1] * 250
            obs[i][2] = obs[i][2] * 30
            obs[i][3] = obs[i][3] * 30
        ego = obs[0]
        for i in range(1,5):
            for j in range(4):
                obs[i][j] += ego[j]
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.config['check_safe']:
            action = self.safety_check(action)
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            print(action)
            print("发生碰撞")
        return obs, reward, terminated, truncated, info

