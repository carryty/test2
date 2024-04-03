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


class MyhighwayEnv(AbstractEnv):
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
            "screen_width": 1920,
            "screen_height":1080,
            "incoming_vehicle_destination": None,
            "spawn_probability": 0.6,
            "lanes_count": 4,
            "vehicles_count": 5,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 5,  # [s]40
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -5,    # The reward received when colliding with a vehicle.    发生碰撞的奖励
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            # "real_time_rendering": True
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10, 10]

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
        net.add_lane("c", "d", StraightLane([120, -20], [120, -42],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -20], [125, -42],
                                            line_types=(LineType.STRIPED, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [105, -42]
        radii2 = 15
        net.add_lane("d", "e",
                     CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-180), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                     CircularLane(center2, radii2 + 5, np.deg2rad(0), np.deg2rad(-180), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[4]))

        # 5 - Vertical Straight
        net.add_lane("e", "f", StraightLane([90, -42], [90, -30],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[3]))
        net.add_lane("e", "f", StraightLane([85, -42], [85, -30],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[3]))

        # 6 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane("f", "g",
                     CircularLane(center3, radii3 + 5, np.deg2rad(0), np.deg2rad(136), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[5]))
        net.add_lane("f", "g",
                     CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[5]))

        # 7 - Slant
        net.add_lane("g", "h", StraightLane([55.7, -15.7], [35.7, -35.7],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("g", "h", StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[6]))

        # 8 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("h", "i",
                     CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                     CircularLane(center4, radii4 + 5, np.deg2rad(315), np.deg2rad(165), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))
        net.add_lane("i", "j",
                     CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5,
                                  clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                  speed_limit=speedlimits[7]))
        net.add_lane("i", "j",
                     CircularLane(center4, radii4 + 5, np.deg2rad(170), np.deg2rad(58), width=5,
                                  clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[7]))

        # 9 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("j", "a",
                     CircularLane(center5, radii5 + 5, np.deg2rad(240), np.deg2rad(270), width=5,
                                  clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                  speed_limit=speedlimits[8]))
        net.add_lane("j", "a",
                     CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
                                  clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                  speed_limit=speedlimits[8]))

        # 10 T字形路口
        lane_width = AbstractLane.DEFAULT_WIDTH  # 4 [m]
        right_turn_radius = 9  # 9 [m]
        # left_turn_radius = right_turn_radius + lane_width  # 13 [m]
        outer_distance = right_turn_radius + lane_width / 2  # 11 [m]
        access_length = 50 + 50  # [m]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        # corner = 3
        # #angle = np.radians(90 * corner)
        # is_horizontal = corner % 2  # 0 1 0 1
        # priority = 3 if is_horizontal else 1
        # rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        # priority = 1
        # Incoming
        start = [200, -33]
        end = [135, -33]
        net.add_lane("wxs", "ir3",
                     StraightLane(start, end, line_types=[s, c], speed_limit=10))
        # Right turn 1
        r_center = [135, -20]
        net.add_lane("c", "il3",
                     CircularLane(r_center, right_turn_radius, np.radians(180), np.radians(270), clockwise=True,
                                  line_types=[n, c], speed_limit=10))
        # Right turn 2
        r_center = [135, -42]
        net.add_lane("ir3", "d",
                     CircularLane(r_center, right_turn_radius, np.radians(90), np.radians(180), clockwise=True,
                                  line_types=[n, c], speed_limit=10))
        # Exit
        start = [135, -29]
        end = [200, -29]
        net.add_lane("il3", "wes",
                     StraightLane(start, end, line_types=[n, c], speed_limit=10))


        ########################################################################################################################环岛
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        center = [240, -31]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        radii = [radius, radius + 4]
        # n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED  # 没有线 实线 虚线
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]车道的振幅5
        delta_st = 0.2 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev  # 车道的脉动频率
        net.add_lane("ser", "ses", StraightLane([2 + center[0], 30], [2 + center[0], dev / 2 + center[1]], line_types=(c, c)))
        net.add_lane("ses", "se",
                     SineLane([2 + a + center[0], dev / 2+ center[1]], [2 + a + center[0], dev / 2 - delta_st + center[1]], a, w, -np.pi / 2, line_types=(c, c)))

        net.add_lane("sx", "sxs", SineLane([-2 - a + center[0], -dev / 2 + delta_en + center[1]], [-2 - a + center[0], dev / 2 + center[1]], a, w, -np.pi / 2 + w * delta_en, line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2 + center[0], dev / 2 + center[1]], [-2 + center[0], access + center[1]-120], line_types=(n, c)))

        # net.add_lane("eer", "ees", StraightLane([access + center[0], -2 + center[1]], [dev / 2 + center[0], -2 + center[1]], line_types=(s, c)))
        # net.add_lane("ees", "ee",
        #              SineLane([dev / 2 + center[0], -2 - a + center[1]], [dev / 2 - delta_st + center[0], -2 - a + center[1]], a, w, -np.pi / 2, line_types=(c, c)))
        # net.add_lane("ex", "exs",
        #              SineLane([-dev / 2 + delta_en + center[0], 2 + a + center[1]], [dev / 2 + center[0], 2 + a + center[1]], a, w, -np.pi / 2 + w * delta_en,
        #                       line_types=(c, c)))
        # net.add_lane("exs", "exr", StraightLane([dev / 2 + center[0], 2 + center[1]], [access + center[0], 2 + center[1]], line_types=(n, c)))

        net.add_lane("ner", "nes", StraightLane([-2 + center[0], -access + center[1]+120], [-2 + center[0], -dev / 2 + center[1]], line_types=(s, c)))

        net.add_lane("nes", "ne", SineLane([-2 - a + center[0], -dev / 2 + center[1]], [-2 - a + center[0], -dev / 2 + delta_st + center[1]], a, w, -np.pi / 2, line_types=(c, c)))

        net.add_lane("nx", "nxs",
                     SineLane([2 + a + center[0], dev / 2 - delta_en + center[1]], [2 + a + center[0], -dev / 2 + center[1]], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2 + center[0], -dev / 2 + center[1]], [2 + center[0], -access + center[1]+100], line_types=(c, c)))

        # net.add_lane("o31", "wes", StraightLane([160, -29], [200, 2 + center[1]], line_types=(s, c)))
        net.add_lane("wes", "we",
                     SineLane([-dev / 2 + center[0], 2 + a + center[1]], [-dev / 2 + delta_st + center[0], 2 + a + center[1]], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wx", "wxs",
                     SineLane([dev / 2 - delta_en + center[0], -2 - a + center[1]], [-dev / 2 + center[0], -2 - a + center[1]], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        # net.add_lane("wxs", "o32", StraightLane([200, -2 + center[1]], [160, -33], line_types=(n, c)))



        #####################################################################################################道路扩展(KZ)

        net.add_lane("nxr", "KZ1", SineLane([242, -101], [242, -111], 4, np.pi/20, 0, line_types=(n, c)))
        # net.add_lane("nxr", "KZ2", SineLane([242, -101], [242, -111], 4, np.pi / 20, -np.pi, line_types=(c, n)))
        net.add_lane("nxr", "lgy", StraightLane([242, -101],[242, -181], line_types=(c, s)))
        net.add_lane("KZ1", "lgy", StraightLane([246, -111], [246, -181], line_types=(n, c)))
        # net.add_lane("KZ2", "lgy0", StraightLane([238, -111], [238, -181], line_types=(c, n)))

        ######################################################################################################弯道1（1_ZW）

        center_1_ZW = [233, -181]
        left_turn_radius_1 = 9  # [m]
        left_turn_radius_2 = 13  # [m]
        net.add_lane("lgy", "1_ZW", CircularLane(center_1_ZW, left_turn_radius_1, np.radians(0), np.radians(-90),clockwise = False, line_types=(c, s)))
        net.add_lane("lgy", "1_ZW",
                     CircularLane(center_1_ZW, left_turn_radius_2, np.radians(0), np.radians(-90), clockwise=False,
                                  line_types=(n, c)))

        #######################################################################################################上直道（Ty）

        net.add_lane("Create_Car_1", "1_ZW", StraightLane([500, -198], [233, -198], line_types=(c, c)))
        net.add_lane("1_ZW", "Ty", StraightLane([233, -198], [-50, -198], line_types=(s, c)))
        net.add_lane("1_ZW", "Ty", StraightLane([233, -194], [-50, -194], line_types=(s, s)))
        net.add_lane("1_ZW", "Ty", StraightLane([233, -190], [-50, -190], line_types=(c, s)))

        #######################################################################################################弯道2（2_ZW）

        center_2_ZW = [-50, -181]
        net.add_lane("Ty", "2_ZW",
                     CircularLane(center_2_ZW, left_turn_radius_1, np.radians(-90), np.radians(-180), clockwise=False,
                                  line_types=(c, s)))
        net.add_lane("Ty", "2_ZW",
                     CircularLane(center_2_ZW, left_turn_radius_2, np.radians(-90), np.radians(-180), clockwise=False,
                                  line_types=(n, c)))


        ########################################################################################################左直道+匝道并入（LZH）

        net.add_lane("Create_Car_2", "2_ZW", StraightLane([-67, -400], [-67, -181], line_types=(c, c)))
        net.add_lane("2_ZW", "LZH_0_1", StraightLane([-67, -181], [-67, -60], line_types=(s, c)))
        net.add_lane("LZH_0_1", "LZH_0_2", StraightLane([-67, -60], [-67, -20], line_types=(s, s)))
        net.add_lane("LZH_0_2", "LZH_0", StraightLane([-67, -20], [-67, 30], line_types=(s, c)))

        net.add_lane("2_ZW", "LZH_0_1", StraightLane([-63, -181], [-63, -60], line_types=(s, s)))
        net.add_lane("LZH_0_1", "LZH_0_2", StraightLane([-63, -60], [-63, -20], line_types=(s, s)))
        net.add_lane("LZH_0_2", "LZH_0", StraightLane([-63, -20], [-63, 30], line_types=(s, s)))

        net.add_lane("2_ZW2", "LZH_2", StraightLane([-59, -181], [-59, -60], line_types=(c, s)))
        net.add_lane("LZH_0_1", "LZH_0_2", StraightLane([-59, -60], [-59, -20], line_types=(c, s)))
        net.add_lane("LZH_0_2", "LZH_0", StraightLane([-59, -20], [-59, 30], line_types=(c, s)))

        net.add_lane("ZaDao_1_start", "ZaDao_1_end", StraightLane([-74.25, -140], [-74.25, -100], line_types=(c, c)))
        net.add_lane("ZaDao_1_end", "LZH_3", SineLane([-74.25, -100], [-74.25, -60], 3.25, np.pi/80, np.radians(180), line_types=(c, c)))
        net.add_lane("LZH_0_1", "LZH_0_2", StraightLane([-71, -60], [-71, -20], line_types=(s, c)))

        ##########################################################################################################弯道3（3_ZW）

        center_3_ZW = [-50, 30]
        net.add_lane("LZH_0", "3_ZW",
                     CircularLane(center_3_ZW, left_turn_radius_1, np.radians(180), np.radians(90), clockwise=False,
                                  line_types=(c, s)))
        net.add_lane("LZH_0", "3_ZW",
                     CircularLane(center_3_ZW, left_turn_radius_2, np.radians(180), np.radians(90), clockwise=False,
                                  line_types=(n, s)))
        net.add_lane("LZH_0", "3_ZW",
                     CircularLane(center_3_ZW, left_turn_radius_2 + 4, np.radians(180), np.radians(90), clockwise=False,
                                  line_types=(n, c)))


        ##########################################################################################################下直道（ZD）

        net.add_lane("3_ZW", "ZD_0", StraightLane([-50, 39], [0, 39], line_types=(c, s)))
        net.add_lane("ZD_0", "ZD_1", StraightLane([0, 39], [233, 39], line_types=(c, s)))
        # net.add_lane("3_ZW", "ZD_2", StraightLane([-50, 39], [233, 39], line_types=(c, s)))

        net.add_lane("3_ZW", "ZD_0", StraightLane([-50, 43], [0, 43], line_types=(n, s)))
        net.add_lane("ZD_0", "ZD_1", StraightLane([0, 43], [233, 43], line_types=(n, c)))
        net.add_lane("ZD_1", "ZD_end", StraightLane([233, 43], [242, 43], line_types=(c, c)))

        net.add_lane("3_ZW", "ZD_0", StraightLane([-50, 47], [0, 47], line_types=(n, c)))
        net.add_lane("ZD_0", "ZaDao_2_end", SineLane([0, 47], [40, 47], 10, np.pi/80, np.radians(0), line_types=(c, c)))
        net.add_lane("ZaDao_2_end", "ZD_2", StraightLane([40, 57], [500, 57], line_types=(c, c)))

        ##########################################################################################################弯道4（4_ZW）

        center_4_ZW = [233, 30]
        net.add_lane("ZD_1", "ser", CircularLane(center_4_ZW, left_turn_radius_1, np.radians(90), np.radians(0), clockwise=False, line_types=(c, c)))



        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])     # Regulated
        road.objects.append(Obstacle(road, [-2 + center[0], -access + center[1]+120]))
        road.objects.append(Obstacle(road, [-2 + center[0], access + center[1]-120]))
        road.objects.append(Obstacle(road, [-50, -198]))
        road.objects.append(Obstacle(road, [-71, -20]))
        road.objects.append(Obstacle(road, [242, 43]))
        self.road = road







        # lane_width = AbstractLane.DEFAULT_WIDTH  # 4 [m]
        # right_turn_radius = lane_width + 5  # 9 [m]
        # left_turn_radius = right_turn_radius + lane_width  # 13 [m]
        # outer_distance = right_turn_radius + lane_width / 2  # 11 [m]
        # access_length = 50 + 50  # [m]
        #
        # net = RoadNetwork()
        # n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        # for corner in range(4):
        #     angle = np.radians(90 * corner)
        #     is_horizontal = corner % 2  # 0 1 0 1
        #     priority = 3 if is_horizontal else 1
        #     rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        #     # Incoming
        #     start = rotation @ np.array([lane_width / 2, access_length + outer_distance])  # @（矩阵乘法）
        #     end = rotation @ np.array([lane_width / 2, outer_distance])
        #     net.add_lane("o" + str(corner), "ir" + str(corner),
        #                  StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
        #     # Right turn
        #     r_center = rotation @ (np.array([outer_distance, outer_distance]))
        #     net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
        #                  CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
        #                               line_types=[n, c], priority=priority, speed_limit=10))
        #     # Left turn
        #     l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
        #     net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
        #                  CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
        #                               clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
        #     # Straight
        #     start = rotation @ np.array([lane_width / 2, outer_distance])
        #     end = rotation @ np.array([lane_width / 2, -outer_distance])
        #     net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
        #                  StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
        #     # Exit
        #     start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
        #     end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
        #     net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
        #                  StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))
        #
        # road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        # self.road = road


        # """Create a road composed of straight adjacent lanes."""
        # self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
        #                  np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])  # 返回IDMVehicle类对象
        # other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        # # print(other_per_controlled)[50]
        # self.controlled_vehicles = []
        # for others in other_per_controlled:
        #     vehicle = Vehicle.lane_create_random(
        #         self.road,
        #         speed=5,
        #         lane_from=["a"],
        #         lane_id=self.config["initial_lane_id"],
        #         spacing=self.config["ego_spacing"]
        #     )
        #     # print(others) 50
        #     vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        #     self.controlled_vehicles.append(vehicle)
        #     self.road.vehicles.append(vehicle)

        # for _ in range(4):
        #     # print(_)
        #     vehicle = other_vehicles_type.create_random(self.road, lane_from="a", speed=8, lane_id=0, spacing=1 / self.config["vehicles_density"])
        #     vehicle.randomize_behavior()
        #     self.road.vehicles.append(vehicle)
        #
        # for _ in range(4):
        #     # print(_)
        #     vehicle = other_vehicles_type.create_random(self.road, lane_from="a", lane_id=1, speed=8, spacing=1 / self.config["vehicles_density"])
        #     vehicle.randomize_behavior()
        #     self.road.vehicles.append(vehicle)

        self.controlled_vehicles = []
        ego_lane = self.road.network.get_lane(("d", "e", 0))
        ego_vehicle = self.action_type.vehicle_class(self.road,
                                                     ego_lane.position(0, 0),
                                                     heading=-90,
                                                     speed=8)
        # try:
        #      ego_vehicle.plan_route_to("ZD_1")
        # except AttributeError:
        #     pass
        self.controlled_vehicles.append(ego_vehicle)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        create_vehicles = self.config["Create_vehicles"]
        if create_vehicles:
            self.road.vehicles.append(
                other_vehicles_type.lane_create_random(self.road, lane_from=["a","b"], speed=15))
            self.road.vehicles.append(
                other_vehicles_type.lane_create_random(self.road, lane_from=["c"], lane_to="d", speed=15))
            self.road.vehicles.append(
                other_vehicles_type.lane_create_random(self.road, lane_from=["e", "f"], speed=15))
            self.road.vehicles.append(
                other_vehicles_type.lane_create_random(self.road, lane_from=["g", "h"], speed=15))
            self.road.vehicles.append(
                other_vehicles_type.lane_create_random(self.road, lane_from=["i", "j"], speed=15))

            # self.road.vehicles.append(
            #     other_vehicles_type(self.road, self.road.network.get_lane(("b", "c", 0)).position(0, 0), speed=15))
            # self.road.vehicles.append(
            #     other_vehicles_type(self.road, self.road.network.get_lane(("f", "g", 0)).position(0, 0), speed=15))
            # self.road.vehicles.append(
            #     other_vehicles_type(self.road, self.road.network.get_lane(("h", "i", 0)).position(0, 0), speed=15))


            self.road.vehicles.append(
                other_vehicles_type(self.road, self.road.network.get_lane(("se", "ex", 0)).position(0, 0), speed=15))
            self.road.vehicles.append(
                other_vehicles_type(self.road, self.road.network.get_lane(("ne", "wx", 0)).position(0, 0), speed=15))
            self.road.vehicles.append(
                other_vehicles_type(self.road, self.road.network.get_lane(("we", "sx", 1)).position(0, 0), speed=17))


        ###########################################################################################################################

        # position_deviation = 2
        # speed_deviation = 2
        #
        # # Ego-vehicle

        #
        # # # Incoming vehicle
        # # destinations = ["031", "nxr"]
        # other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        # vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                            ("we", "sx", 1),
        #                                            longitudinal=5 + self.np_random.normal() * position_deviation,
        #                                            speed=16 + self.np_random.normal() * speed_deviation)
        #
        # if self.config["incoming_vehicle_destination"] is not None:
        #     destination = destinations[self.config["incoming_vehicle_destination"]]
        # else:
        #     destination = self.np_random.choice(destinations)
        # vehicle.plan_route_to(destination)
        # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)
        #
        # # Other vehicles
        # for i in list(range(1, 2)) + list(range(-1, 0)):
        #     vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                                ("we", "sx", 0),
        #                                                longitudinal=20 * i + self.np_random.normal() * position_deviation,
        #                                                speed=16 + self.np_random.normal() * speed_deviation)
        #     vehicle.plan_route_to(self.np_random.choice(destinations))
        #     vehicle.randomize_behavior()
        #     self.road.vehicles.append(vehicle)
        #
        # Entering vehicle
        # time.sleep(5)
        # vehicle = other_vehicles_type.make_on_lane(self.road,
        #                                            ("a", "b", 0),
        #                                            longitudinal=50 + self.np_random.normal() * position_deviation,
        #                                            speed=16)
        # vehicle.plan_route_to("a")
        # vehicle.randomize_behavior()
        # self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        # self._clear_vehicles()
        # self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, truncated, info

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        # route = self.np_random.choice(range(4), size=2, replace=False)
        # route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("Create_Car_1", "1_ZW", 0),
                                            longitudinal= 260,
                                            speed=20)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("ZD_2")
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle
