import random
from typing import Dict, Text
import math
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class HighwayEnv(AbstractEnv):
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
                "absolute": True,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],  #,"Lane_type"
                "features_range": {"x": [-250, 250], "y": [-250, 250], "vx": [-30, 30], "vy": [-30, 30]},
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,   #4
            "vehicles_count": 50,   #50
            "controlled_vehicles": 1,   #1
            "initial_lane_id": None,    #None
            "duration": 80,  # [s]40
            "ego_spacing": 2,   #2
            "vehicles_density": 1,  #1
            "collision_reward": -10,    # 与车辆相撞时获得的奖励
            "right_lane_reward": 0.1,  # 在最右侧车道上行驶时获得的奖励，其他车道线性映射为零。0.1
            "high_speed_reward": 0.3,  # 速行驶时获得的奖励，根据配置[“reward_speed_range”]，低速时线性映射为零
            "low_speed_reward":-0.1,
            "lane_change_reward": 0,   # 每次变道时获得的奖励。
            "reward_speed_range": [20, 30],
            "reward_low_speed_range": [10, 16],
            "normalize_reward": False,
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """创建一条由直线相邻车道组成的道路"""
        angles = random.choice([0, math.pi/2, math.pi, math.pi*3/2])
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"],angle=angles, speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """创建一些给定类型的新随机车辆，并将它们添加到道路上。"""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=20,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"], speed=20)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        该奖励旨在促进在最右侧车道上高速行驶，并避免碰撞。
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"]+self.config["low_speed_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed # * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        low_scaled_speed = utils.lmap(forward_speed, self.config["reward_low_speed_range"], [1, 0])
        return {
            "collision_reward": float(self.vehicle.crashed),
            # "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "low_speed_reward":np.clip(low_scaled_speed,0,1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """如果ego车辆发生碰撞，这一集就结束了"""
        # print(self.controlled_vehicles[0].to_dict()['vx'], self.controlled_vehicles[0].to_dict()['vy'])
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """如果达到时间限制，该集将被截断。"""
        return self.time >= self.config["duration"]

    def safety_check(self, action: int,obs):
        obs = np.delete(obs, 0, axis=1)
        print(obs)
        # print(self.controlled_vehicles[0].to_dict()['heading'])
        # print(type(obs))
        # print("start")
        # for _ in obs:
        #     print(_)
        # print("end")

    # def _is_safe(self, action, obs, heading):


    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        _obs = self.observation_type.observe()
        print(_obs)
        # action_test =
        self.safety_check(action,_obs)
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info


class HighwayEnvFast(HighwayEnv):
    """
    highway-v0的一种变体，执行速度更快：
        - 较低的模拟频率
        - 场景中的车辆更少（车道更少，剧集持续时间更短）
        - 仅检查受控车辆与其他车辆的碰撞
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        #禁用未控制车辆的碰撞检查
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
