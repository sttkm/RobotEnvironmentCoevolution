import math
import numpy as np

from gym import error, spaces

from evogym import EvoWorld, WorldObject
from evogym.utils import *
from evogym.envs import BenchmarkBase

from gym.envs.registration import register

register(
    id = 'Parkour-v0',
    entry_point = 'custom_envs.parkour:Parkour',
    max_episode_steps=512
)

register(
    id = 'Parkour-v1',
    entry_point = 'custom_envs.parkour:ParkourFlip',
    max_episode_steps=512
)

class Parkour(BenchmarkBase):

    def __init__(self, robot, terrain):
        body, connections = robot

        # make world
        self.world = self.build(terrain)
        self.world.add_from_array('robot', body, 1, terrain['start_height']+1, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 10

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)

        # terrain
        self.terrain_list = list(terrain['objects'].keys())

    def build(self, terrain):
        world = EvoWorld()

        file_grid_size = Pair(terrain['grid_width'], terrain['grid_height'])

        # read in objects
        for name, obj_data in terrain['objects'].items():

            obj = WorldObject()
            obj.load_from_parsed_json(name, obj_data, file_grid_size)
            world.add_object(obj)

        return world


    def step(self, action):

        # collect pre step information
        robot_pos_init = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        robot_pos_final = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))
       
        # compute reward
        robot_com_pos_init = np.mean(robot_pos_init, axis=1)
        robot_com_pos_final = np.mean(robot_pos_final, axis=1)
        reward = (robot_com_pos_final[0] - robot_com_pos_init[0])
        
        # error check unstable simulation
        if done:
            # print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check if y coordinate is below lowest platform
        under = np.min(robot_pos_final[1,:])
        if under < (0.5)*self.VOXEL_SIZE:
            reward -= max(min(robot_com_pos_init[0], 5.0), 3.0)
            done = True
        # if com[1] < (5)*self.VOXEL_SIZE:
        #     reward -= 3.0
        #     done = True

        if robot_com_pos_final[0] > (self.world.grid_size.x)*self.VOXEL_SIZE:
            done = True

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))

        return obs



class ParkourFlip(BenchmarkBase):

    def __init__(self, robot, connections, terrain):
        body, connections = robot

        # make world
        self.world = self.build(terrain)
        self.world.add_from_array('robot', body, 1, terrain['start_height']+1, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 10

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        # self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(3 + num_robot_points + (2*self.sight_dist +1),), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + 2*num_robot_points + (2*self.sight_dist +1),), dtype=np.float)
        # print(self.object_vel_at_time(self.get_time(), "robot").shape)

        # terrain
        self.terrain_list = list(terrain['objects'].keys())

        self.num_flips = 0

    def build(self, terrain):
        world = EvoWorld()

        file_grid_size = Pair(terrain['grid_width'], terrain['grid_height'])

        # read in objects
        for name, obj_data in terrain['objects'].items():

            obj = WorldObject()
            obj.load_from_parsed_json(name, obj_data, file_grid_size)
            world.add_object(obj)

        return world


    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        ort_1 = self.object_orientation_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")
        ort_2 = self.object_orientation_at_time(self.get_time(), "robot")

        # observation
        ort = self.get_ort_obs("robot")
        obs = np.concatenate((
            # self.get_vel_com_obs("robot"),
            self.object_vel_at_time(self.get_time(), "robot").flatten(),
            np.sin(ort),
            np.cos(ort),
            # self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))
       
        # compute reward
        com_1 = np.mean(pos_1, axis=1)
        com_2 = np.mean(pos_2, axis=1)
        move = (com_2[0] - com_1[0])

        # update flips
        flattened_ort_1 = self.num_flips + ort_1 / (2 * math.pi)
        # flattened_ort_1 = self.num_flips * 2 * math.pi + ort_1

        if ort_1 < math.pi/3 and ort_2 > 5*math.pi/3:
            self.num_flips -= 1
        if ort_1 > 5*math.pi/3 and ort_2 <  math.pi/3:
            self.num_flips += 1

        # if self.num_flips > -10:
        flattened_ort_2 = self.num_flips + ort_2 / (2 *  math.pi)
        # flattened_ort_2 = self.num_flips * 2 * math.pi + ort_2
        rotate = -(flattened_ort_2 - flattened_ort_1)

        reward = rotate if move>0 else -abs(rotate)
        reward += move if rotate>0 else -abs(move)

        # error check unstable simulation
        if done:
            reward -= -self.num_flips
            # reward -= 3.0

        # check if y coordinate is below lowest platform
        if com_2[1] < (5)*self.VOXEL_SIZE:
            # reward -= -self.num_flips
            reward -= max(min(com_1[0], 5.0), 3.0) + self.num_flips * 0.5
            # reward -= 3.0
            done = True

        if com_2[0] > (self.world.grid_size.x)*self.VOXEL_SIZE:
            done = True

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        self.num_flips = 0

        # observation
        ort = self.get_ort_obs("robot")
        obs = np.concatenate((
            # self.get_vel_com_obs("robot"),
            self.object_vel_at_time(self.get_time(), "robot").flatten(),
            np.sin(ort),
            np.cos(ort),
            # self.get_ort_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
            ))

        return obs


# class ParkourTraining(Parkour):
#     def __init__(self, robot, terrain):
#         assert terrain['type'] in ["step", "soft", "horl"]
#         self.robot = robot
#         self.terrain_type = terrain['type']
#         self.terrain_width = terrain['width']
        
#         self.build()

#     def build(self):
#         if self.terrain_type == "step":
#             x = 10
#             y = 0
#             while True:
#                 step = np.random.uniform(-1,1)
#                 width = np.random.
#             {"grid_width": 100, "grid_height": 7, "start_height": 7, "objects": {"platfotm1": {"types": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], "indices": [700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799], "neighbors": {"700": [701], "701": [700, 702], "702": [701, 703], "703": [702, 704], "704": [703, 705], "705": [704, 706], "706": [705, 707], "707": [706, 708], "708": [707, 709], "709": [708, 710], "710": [709, 711], "711": [710, 712], "712": [711, 713], "713": [712, 714], "714": [713, 715], "715": [714, 716], "716": [715, 717], "717": [716, 718], "718": [717, 719], "719": [718, 720], "720": [719, 721], "721": [720, 722], "722": [721, 723], "723": [722, 724], "724": [723, 725], "725": [724, 726], "726": [725, 727], "727": [726, 728], "728": [727, 729], "729": [728, 730], "730": [729, 731], "731": [730, 732], "732": [731, 733], "733": [732, 734], "734": [733, 735], "735": [734, 736], "736": [735, 737], "737": [736, 738], "738": [737, 739], "739": [738, 740], "740": [739, 741], "741": [740, 742], "742": [741, 743], "743": [742, 744], "744": [743, 745], "745": [744, 746], "746": [745, 747], "747": [746, 748], "748": [747, 749], "749": [748, 750], "750": [749, 751], "751": [750, 752], "752": [751, 753], "753": [752, 754], "754": [753, 755], "755": [754, 756], "756": [755, 757], "757": [756, 758], "758": [757, 759], "759": [758, 760], "760": [759, 761], "761": [760, 762], "762": [761, 763], "763": [762, 764], "764": [763, 765], "765": [764, 766], "766": [765, 767], "767": [766, 768], "768": [767, 769], "769": [768, 770], "770": [769, 771], "771": [770, 772], "772": [771, 773], "773": [772, 774], "774": [773, 775], "775": [774, 776], "776": [775, 777], "777": [776, 778], "778": [777, 779], "779": [778, 780], "780": [779, 781], "781": [780, 782], "782": [781, 783], "783": [782, 784], "784": [783, 785], "785": [784, 786], "786": [785, 787], "787": [786, 788], "788": [787, 789], "789": [788, 790], "790": [789, 791], "791": [790, 792], "792": [791, 793], "793": [792, 794], "794": [793, 795], "795": [794, 796], "796": [795, 797], "797": [796, 798], "798": [797, 799], "799": [798]}}}}
#             terrain = []
#         elif self.terrain_type == "soft":
#             terrain = []
#         else:
#             terrain = []
        
#         super().__init__(self.robot, terrain)

#     def reset(self):
        
#         super().reset()

#         self.build()

#         # observation
#         obs = np.concatenate((
#             self.get_vel_com_obs("robot"),
#             self.get_ort_obs("robot"),
#             self.get_relative_pos_obs("robot"),
#             self.get_floor_obs("robot", self.terrain_list, self.sight_dist),
#             ))

#         return obs