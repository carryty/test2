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


class CircularLaneEnv(AbstractEnv):
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
            "screen_width": 1880,
            "screen_height":1100,
            "incoming_vehicle_destination": None,
            "spawn_probability": 0.6,
            "lanes_count": 2,
            "vehicles_count": 5,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 90,  # [s]40
            "ego_spacing": 2,
            "vehicles_density": 1,
            # "b_reward": 10,
            # "d_reward": 10,
            # "f_reward": 10,
            # "h_reward": 10,
            "collision_reward": -55,    # The reward received when colliding with a vehicle.    发生碰撞的奖励
            # "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 5,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "low_speed_reward": -2,
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "low_reward_speed_range": [10, 16],
            "normalize_reward": False,
            "offroad_terminal": True,
            # "real_time_rendering": True,
            "show_point": False,
            "simulation_frequency": 80,  # [Hz]50
            "policy_frequency": 1,  # [Hz]3
            "is_truncated": True,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        self.t = 0
        self.cls = 0
        # self.Arrival = [1,1,1,1]

    def _create_road(self) -> None:
        net = RoadNetwork()

        r1 = 15
        r2 = 95

        # 弯道1
        center1 = [-80, 0]
        net.add_lane("a", "b",
                     CircularLane(center1, r1 - 5, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("a", "b",
                     CircularLane(center1, r1, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("a", "b",
                     CircularLane(center1, r1 + 5, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("a", "b",
                     CircularLane(center1, r1 + 10, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=30))

        # 弯道2
        center2 = [-40, 0]
        net.add_lane("b", "c",
                     CircularLane(center2, r1 + 15, np.deg2rad(-180), np.deg2rad(2), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=30))
        net.add_lane("b", "c",
                     CircularLane(center2, r1 + 10, np.deg2rad(-180), np.deg2rad(2), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.STRIPED),
                                  speed_limit=30))
        net.add_lane("b", "c",
                     CircularLane(center2, r1 + 5, np.deg2rad(-180), np.deg2rad(2), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.STRIPED),
                                  speed_limit=30))
        net.add_lane("b", "c",
                     CircularLane(center2, r1, np.deg2rad(-180), np.deg2rad(2), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=30))
        # 弯道3
        center3 = [0, 0]
        net.add_lane("c", "d",
                     CircularLane(center3, r1 - 5, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("c", "d",
                     CircularLane(center3, r1, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("c", "d",
                     CircularLane(center3, r1 + 5, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("c", "d",
                     CircularLane(center3, r1 + 10, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=30))
        # 弯道4
        center4 = [40, 0]
        net.add_lane("d", "e",
                     CircularLane(center4, r1 + 15, np.deg2rad(-180), np.deg2rad(2), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=30))
        net.add_lane("d", "e",
                     CircularLane(center4, r1 + 10, np.deg2rad(-180), np.deg2rad(2), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.STRIPED),
                                  speed_limit=30))
        net.add_lane("d", "e",
                     CircularLane(center4, r1 + 5, np.deg2rad(-180), np.deg2rad(2), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.STRIPED),
                                  speed_limit=30))
        net.add_lane("d", "e",
                     CircularLane(center4, r1, np.deg2rad(-180), np.deg2rad(2), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=30))
        # 弯道5
        center5 = [80, 0]
        net.add_lane("e", "f",
                     CircularLane(center5, r1 - 5, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("e", "f",
                     CircularLane(center5, r1, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("e", "f",
                     CircularLane(center5, r1 + 5, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("e", "f",
                     CircularLane(center5, r1 + 10, np.deg2rad(180), np.deg2rad(-1), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=30))
        # 弯道6
        net.add_lane("f", "a",
                     CircularLane(center3, r2 - 5, np.deg2rad(0), np.deg2rad(-180), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("f", "a",
                     CircularLane(center3, r2, np.deg2rad(0), np.deg2rad(-180), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("f", "a",
                     CircularLane(center3, r2 + 5, np.deg2rad(0), np.deg2rad(-180), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.NONE),
                                  speed_limit=30))
        net.add_lane("f", "a",
                     CircularLane(center3, r2 + 10, np.deg2rad(0), np.deg2rad(-180), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=30))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        # road.objects.append(Obstacle(road, [18.1, 8]))
        # road.objects.append(Obstacle(road, [18.1, 11]))
        self.road = road


    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])  # 返回IDMVehicle类对象

        self.controlled_vehicles = []
        ego_lane = self.road.network.get_lane(("f", "a", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(0, 0),
                                                     heading=0,
                                                     speed=30)

        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        # print(ego_vehicle.to_dict())
        create_vehicles = self.config["Create_vehicles"]
        if create_vehicles:
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["a"], speed=9))
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["b"], speed=12))
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["c"], speed=15))
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["d"], speed=17))
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["e"], speed=19))
            # self.road.vehicles.append(
            #     other_vehicles_type.lane_create_random(self.road, lane_from=["f"],speed=19))

            # other_lane = self.road.network.get_lane(("f", "a", 0))
            # other1_vehicle = self.action_type.vehicle_class(self.road,
            #                                                 other_lane.position(0, 0),
            #                                                 heading=0,
            #                                                 speed=20)
            # other1_vehicle = create_random(self.road, speed = 19, lane_from = f, lane_to = a)

            # self.road.vehicles.append(
            #     other_vehicles_type.create_random(self.road, speed = 19, lane_from = "f", lane_to = "a",spacing = 1.5).randomize_behavior())


            # v = other_vehicles_type.create_random(self.road, speed = 19, lane_from = "f", lane_to = "a",lane_id=0,spacing = 1.2)
            # v.randomize_behavior()
            # self.road.vehicles.append(v)
            #
            # v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a",lane_id=0, spacing=1.2)
            # v.randomize_behavior()
            # self.road.vehicles.append(v)
            #
            # v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a", lane_id=0,
            #                                       spacing=1.2)
            # v.randomize_behavior()
            # self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a",lane_id=1, spacing=1.4)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a",lane_id=1, spacing=1.4)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a", lane_id=1,
                                                  spacing=1.4)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a",lane_id=2, spacing=1.4)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a", lane_id=2 ,spacing=1.4)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a", lane_id=2,
                                                  spacing=1.4)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a", lane_id=3,spacing=1.4)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a", lane_id=3, spacing=1.4)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a", lane_id=3,spacing=1.4)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="b",lane_id=0, spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="b", lane_id=2, spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="b", lane_id=3, spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="a", lane_id=1, spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=15, lane_from="a", lane_id=1, spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="c", lane_id=3,spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="d", lane_id=0,spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="d", lane_id=3,spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="d", lane_id=1, spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            v = other_vehicles_type.create_random(self.road, speed=19, lane_from="c", lane_id=2, spacing=3.5)
            v.randomize_behavior()
            self.road.vehicles.append(v)

            # v = other_vehicles_type.create_random(self.road, speed=19, lane_from="d", lane_id=3,spacing=3)
            # v.randomize_behavior()
            # self.road.vehicles.append(v)
            #
            # v = other_vehicles_type.create_random(self.road, speed=19, lane_from="d",lane_id=0, spacing=3)
            # v.randomize_behavior()
            # self.road.vehicles.append(v)

            # self.road.vehicles.append(
            #     other_vehicles_type.create_random(self.road, speed=16, lane_from="f", lane_to="a", spacing = 1.5).randomize_behavior())
            # self.road.vehicles.append(
            #     other_vehicles_type.create_random(self.road, speed=15, lane_from="f", lane_to="a", spacing = 1.5).randomize_behavior())
            # self.road.vehicles.append(
            #     other_vehicles_type.create_random(self.road, speed=19, lane_from="f", lane_to="a", spacing = 1.5).randomize_behavior())
            # self.road.vehicles.append(
            #     other_vehicles_type.create_random(self.road, speed=17, lane_from="f", lane_to="a", spacing=1.5).randomize_behavior())
            # self.road.vehicles.append(
            #     other_vehicles_type.create_random(self.road, speed=19, lane_from="a", lane_to="b", spacing=1.5).randomize_behavior())
            # self.road.vehicles.append(
            #     other_vehicles_type.create_random(self.road, speed=13, lane_from="b", lane_to="c", ).randomize_behavior())
            # self.road.vehicles.append(
            #     other_vehicles_type.create_random(self.road, speed=19, lane_from="c", lane_to="d",).randomize_behavior())
            # self.road.vehicles.append(
            #     other_vehicles_type.create_random(self.road, speed=16, lane_from="d", lane_to="e", ).randomize_behavior())


        # for v in self.road.vehicles:
        #     print(v.to_dict())

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        # print(rewards)
        # print("                 ")
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = reward - (self.time - self.t) * 0.6  #0.8
        self.t = self.time
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        # self.point_rewards()
        # print(reward)
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        # print(self.vehicle.speed)
        forward_speed = self.vehicle.speed     # * np.cos(self.vehicle.heading)
        self.cls+=1
        if self.cls%30 == 0:
            print(forward_speed)
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
