import time
from typing import Dict, Text, Tuple
from gym import register
import numpy as np

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


class NewRacetrackEnv(AbstractEnv):
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
                "features_range": {"x": [-200, 200], "y": [-200, 200], "vx": [-30, 30], "vy": [-30, 30]},
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
            "vehicles_count": 5,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]40
            "ego_spacing": 2,
            "vehicles_density": 1,
            "b_reward": 15,
            "d_reward": 20,
            "f_reward": 25,
            "h_reward": 50,
            "collision_reward": -60,    # The reward received when colliding with a vehicle.    发生碰撞的奖励
            # "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.8,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "low_speed_reward": -0.8,
            "reward_speed_range": [10, 15],
            "reward_low_speed_range": [1, 5], #
            "normalize_reward": False,
            "offroad_terminal": True,
            # "real_time_rendering": True,
            "show_point": True,
            "simulation_frequency": 50,  # [Hz]50
            "policy_frequency": 3,  # [Hz]5
            "is_truncated": True,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.t=0
        self.Arrival = [1, 1, 1, 1]

    def _create_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                            speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b",
                     StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                  speed_limit=speedlimits[1]))

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                     CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                     CircularLane(center1, radii1 + 5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[2]))

        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([120, -20], [120, -30],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -20], [125, -30],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2 + 5, np.deg2rad(0), np.deg2rad(-181), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane("e", "f",
                     CircularLane(center3, radii3 + 5, np.deg2rad(0), np.deg2rad(136), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                     CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 6 - Slant
        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[6]))

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("g", "h",
                     CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                     CircularLane(center4, radii4 + 5, np.deg2rad(315), np.deg2rad(165), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4 + 5, np.deg2rad(170), np.deg2rad(58), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("i", "a",
                     CircularLane(center5, radii5 + 5, np.deg2rad(240), np.deg2rad(270), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        # road.objects.append(Obstacle(road, [18.1, 8]))
        # road.objects.append(Obstacle(road, [18.1, 11]))
        self.road = road

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])  # 返回IDMVehicle类对象

        self.controlled_vehicles = []
        ego_lane = self.road.network.get_lane(("a", "b", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(0, 0),
                                                     heading=0,
                                                     speed=8)
        try:
             ego_vehicle.plan_route_to("i")
        except AttributeError:
            pass
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        create_vehicles = self.config["Create_vehicles"]
        if create_vehicles:
            self.road.vehicles.append(
                other_vehicles_type.lane_create_random(self.road, lane_from=["b", "c"], speed=9))
            self.road.vehicles.append(
                other_vehicles_type.lane_create_random(self.road, lane_from=["e"], speed=9))
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["f", "g"], speed=9))
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["c","d"], speed=9))
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["e", "f"], speed=9))
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["g", "h"], speed=9))


    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = reward + self.point_rewards()
        # t = self.time
        # print(reward)

        reward = reward - (self.time-self.t)*0.7
        # print(reward)
        self.t = self.time
        # print(self.t)
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"]+self.config["low_speed_reward"],
                                 self.config["high_speed_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        # self.point_rewards()
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        low_scaled_speed = utils.lmap(forward_speed, self.config["reward_low_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "low_speed_reward": np.clip(low_scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def point_rewards(self) -> float:
        b = [100, 5]
        d = [125, -30]
        f = [59.4, -19.2]
        h = [-6.9, -18.1]
        # print(self.controlled_vehicles[0].to_dict()['x'], self.controlled_vehicles[0].to_dict()['y'])
        vehicle_x = self.controlled_vehicles[0].to_dict()['x']
        vehicle_y = self.controlled_vehicles[0].to_dict()['y']
        if -3 < vehicle_x - b[0] < 3 and -6 <= vehicle_y - b[1] <= 6 and self.Arrival[0]:
            self.Arrival[0] = 0
            if self.config["show_point"]:
                print("经过b点")
            return self.config["b_reward"]
        elif -6 <= vehicle_x - d[0] <= 6 and -5 < vehicle_y - d[1] < 5 and self.Arrival[1]:
            self.Arrival[1] = 0
            if self.config["show_point"]:
                print("经过d点")
            return self.config["d_reward"]
        elif -4 < vehicle_x - f[0] < 4 and -5 < vehicle_y - f[1] < 5 and self.Arrival[2]:
            self.Arrival[2] = 0
            if self.config["show_point"]:
                print("经过f点")
            return self.config["f_reward"]
        elif -4 < vehicle_x - h[0] < 4 and -4 < vehicle_y - h[1] < 4 and self.Arrival[3]:
            self.Arrival[3] = 0
            if self.config["show_point"]:
                print("经过h点")
            return self.config["h_reward"]
        else:
            return 0


    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        if self.config["is_truncated"]:
            vehicle_x = self.controlled_vehicles[0].to_dict()['x']
            vehicle_y = self.controlled_vehicles[0].to_dict()['y']
            if -6 < vehicle_x - 18.1 < 6 and -6 < vehicle_y - 8.9 < 6:
                print("经过i点")
                self.Arrival = [1, 1, 1, 1]
            # return  (-4<vehicle_x-18.1<4 and -4<vehicle_y-8.9<4)
        return self.time >= self.config["duration"]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # print(self.controlled_vehicles[0].to_dict()['x'])
        obs, reward, terminated, truncated, info = super().step(action)
        # self._clear_vehicles()
        # self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    # def _spawn_vehicle(self,
    #                    longitudinal: float = 0,
    #                    position_deviation: float = 1.,
    #                    speed_deviation: float = 1.,
    #                    spawn_probability: float = 0.6,
    #                    go_straight: bool = False) -> None:
    #     if self.np_random.uniform() > spawn_probability:
    #         return
    #
    #     # route = self.np_random.choice(range(4), size=2, replace=False)
    #     # route[1] = (route[0] + 2) % 4 if go_straight else route[1]
    #     vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
    #     vehicle = vehicle_type.make_on_lane(self.road, ("Create_Car_1", "1_ZW", 0),
    #                                         longitudinal= 260,
    #                                         speed=20)
    #     for v in self.road.vehicles:
    #         if np.linalg.norm(v.position - vehicle.position) < 15:
    #             return
    #     vehicle.plan_route_to("ZD_2")
    #     vehicle.randomize_behavior()
    #     self.road.vehicles.append(vehicle)
    #     return vehicle
