import numpy as np

from .data import get_data


class UserEquipment:
    def __init__(self, num_channels, num_points, possion_lambda, pmax, dmin, dmax, net='resnet152', test=False):
        self.num_channels = num_channels
        self.num_points = num_points
        self.possion_lambda = possion_lambda
        self.pmax = pmax
        self.dmin = dmin
        self.dmax = dmax
        self.net = net
        self.test = test
        self.reset()

    def reset(self):
        # 初始化位置、随机选择action、统计数据
        self.locate_init()
        self.action_init()
        self.statistic_init()

    def locate_init(self):
        # self.distance = 50
        if self.test:
            self.distance = 50
        else:
            self.distance = (np.random.random() * (self.dmax - self.dmin)) + self.dmin

    def action_init(self):
        if self.test:
            self.point = 49
            self.channel = 0
            self.power = 1e-10
        else:
            self.point = np.random.choice(np.arange(self.num_points))
            self.channel = np.random.choice(np.arange(self.num_channels))
            self.power = np.random.random() * (self.pmax - 1e-10) + 1e-10
        self.next_point = self.point
        self.next_channel = self.channel

    def statistic_init(self):
        self.time_used = 0
        self.energy_used = 0
        self.finished_num = 0

        self.head_time_used = 0
        self.head_energy_used = 0
        self.tail_time_used = 0
        self.tail_energy_used = 0

    def finish_task(self):
        self.finished_num += 1
        self.left_task_num -= 1
        if self.left_task_num != 0:
            self.point = self.next_point
            self.channel = self.next_channel
            self.start_task()
        else:
            self.free()

    def receive_tasks(self):
        if self.test:
            self.left_task_num = self.possion_lambda
        else:
            self.left_task_num = np.random.poisson(self.possion_lambda)
        self.left_task_num = max(1, self.left_task_num)
        self.total_task = self.left_task_num

    def start_task(self):
        '''start a new task'''
        if self.left_task_num == 0:
            raise RuntimeError('No tasks left')
        
        mid_data_size, head_latency, head_power, tail_latency, tail_power = get_data(self.net, self.point)
        self.head_time_left = head_latency
        self.tail_time_left = tail_latency
        self.time_left = self.head_time_left + self.tail_time_left
        self.data_left = mid_data_size

        self.local_power = head_power
        self.mec_power = tail_power
        if self.in_cloud_mode():
            self.offloading()
            self.inference_power = 0
        elif self.in_local_mode():
            self.inferring()
            self.inference_power = self.local_power
        else:
            self.inferring()
            self.inference_power = self.mec_power

    def inferring(self):
        self.is_inferring = True
        self.is_offloading = False
        self.is_free = False

    def offloading(self):
        self.is_inferring = False
        self.is_offloading = True
        self.is_free = False

    def free(self):
        self.is_inferring = False
        self.is_offloading = False
        self.is_free = True

    def in_local_mode(self):
        if self.point == 49:
            return True
        else:
            return False

    def in_cloud_mode(self):
        if self.point == 0:
            return True
        else:
            return False

    def in_mec_mode(self):
        if self.in_local_mode() or self.in_cloud_mode():
            return False
        else:
            return True
