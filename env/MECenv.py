import numpy as np

from .channel import SubChannel
from .user import UserEquipment


normalize_factor = {'vgg11': (0.092876, 1204224),
                    'resnet18': (0.045887, 1204224),
                    'resnet152': (0.8423508167266845, 3212444),
                    'mobilenetv2': (0.052676, 1204224)}


class MECsystem(object):
    def __init__(self, slot_time, num_users, num_channels, user_params, channel_params, beta=0.5):
        self.slot_time = slot_time
        self.num_users = num_users

        self.UEs = [UserEquipment(**user_params) for _ in range(num_users)]
        self.channels = [SubChannel(**channel_params) for _ in range(num_channels)]

        self.beta = beta

    def get_state(self):
        state = []
        for u in self.UEs:
            max_time, max_data = normalize_factor[u.net]
            # 完成任务数、剩余时间、剩余数据、距离比
            state.append(u.left_task_num / u.total_task)
            state.append((u.head_time_left + u.tail_time_left) / max_time)
            state.append(u.data_left / max_data)
            state.append((u.distance - u.dmin) / (u.dmax - u.dmin))
        return state

    def get_reward(self):
        energy = np.mean([u.energy_used for u in self.UEs])
        finished = np.mean([u.finished_num for u in self.UEs])
        time = np.mean([u.time_used for u in self.UEs])
        avg_e = energy / max(finished, 0.8)
        avg_t = time / max(finished, 0.8)
        reward = - avg_t / 0.8312 - self.beta * avg_e / 15.64466
        # reward = - avg_t - self.beta * avg_e
        return reward
    
    def get_divided_reward(self):
        rewards = []
        for u in self.UEs:
            energy = u.energy_used
            finished = u.finished_num
            time = u.time_used
            # avg_e = energy / max(finished, 0.8)
            # avg_t = time / max(finished, 0.8)
            avg_t = time
            avg_e = energy
            reward = - avg_t / 0.8312 - self.beta * avg_e / 124.2
            # reward = - avg_t
            # reward = - avg_t - self.beta * avg_e
            rewards.append(reward)
        return rewards

    def reset(self):
        self.time = 0
        for u in self.UEs:
            u.reset()
            u.receive_tasks()
            if u.left_task_num != 0:
                u.start_task()
        for c in self.channels:
            c.reset()
        return self.get_state()

    def step(self, action):
        # init
        done = False
        time_in_slot = 0
        # 一开始传入的完整的本地执行
        # 此时传入输出的action
        self.assign_action(action)

        for u in self.UEs:
            u.statistic_init()

        # state step
        next_time = self.stationary_time()
        # 执行下次决策还有剩余时间时
        while time_in_slot + next_time < self.slot_time:
            self.slot_step(next_time)
            time_in_slot += next_time
            next_time = self.stationary_time()
        # 当当前时隙还有剩余时
        if self.slot_time - time_in_slot > 0:
            self.slot_step(self.slot_time - time_in_slot)

        # done?
        self.time += self.slot_time
        if self.is_done():
            done = True

        # state & reward
        state = self.get_state()
        reward = self.get_divided_reward()

        # info
        total_time_used = sum([u.time_used for u in self.UEs])
        # total_time_used = self.slot_time * self.num_users
        total_energy_used = sum([u.energy_used for u in self.UEs])
        total_finished = sum([u.finished_num for u in self.UEs])
        info = {'total_time_used': total_time_used,
                'total_energy_used': total_energy_used,
                'total_finished': total_finished}

        return state, reward, done, info


    def slot_step(self, time):
        for u in self.UEs:
            if u.is_inferring:
                u.time_used += time
                u.energy_used += u.inference_power * time
                if (u.head_time_left > 0):
                    if (u.head_time_left - time) < 1e-10:
                        u.head_time_left = 0
                        u.inference_power = u.mec_power
                        u.offloading()
                    elif u.head_time_left > time:
                        u.head_time_left -= time
                    else:
                        raise RuntimeError(f'left head inference time {u.head_time_left}s < step time {time}s')
                    
                elif (u.tail_time_left > 0):
                    if (u.tail_time_left - time) < 1e-10:
                        u.tail_time_left = 0
                        u.finish_task()
                    elif u.tail_time_left > time:
                        u.tail_time_left -= time
                    else:
                        raise RuntimeError(f'left tail inference time {u.tail_time_left}s < step time {time}s')
                    
                else:
                    raise RuntimeError('user has no slot inferring time left')
                    
                # if u.in_local_mode():
                #     if (u.head_time_left - time) < 1e-10:
                #         u.head_time_left = 0
                #         u.finish_task()
                #     elif u.head_time_left > time:
                #         u.head_time_left -= time
                #     else:
                #         raise RuntimeError(f'left inference time {u.time_left}s < step time {time}s')
                # elif u.in_mec_mode():
                #     if u.head_time_left > 0:
                #         if (u.head_time_left - time) < 1e-10:
                #             u.head_time_left = 0
                #             u.inference_power = u.mec_power
                #             u.offloading()
                #         elif u.head_time_left > time:
                #             u.head_time_left -= time
                #         else:
                #             raise RuntimeError(f'left head inference time {u.head_time_left}s < step time {time}s')
                #     else:
                #         if (u.tail_time_left - time) < 1e-10:
                #             u.tail_time_left = 0
                #             u.finish_task()
                #         elif u.tail_time_left > time:
                #             u.tail_time_left -= time
                #         else:
                #             raise RuntimeError(f'left tail inference time {u.tail_time_left}s < step time {time}s')
                # else:
                #     raise RuntimeError('enter local inference in cloud mode')

            elif u.is_offloading:
                u.time_used += time
                u.energy_used += u.power * time
                if (u.data_left / u.uplink_rate - time) < 1e-10:
                    # -> inferring, offloading, or free
                    u.data_left = 0
                    u.inference_power = u.mec_power
                    u.inferring()
                    # if u.in_mec_mode():
                    #     u.inference_power = u.mec_power
                    #     u.inferring()
                    # else:
                    #     u.finish_task()
                elif u.data_left / u.uplink_rate > time:
                    u.data_left -= u.uplink_rate * time
                else:
                    raise RuntimeError(f'left offloading time {u.data_left / u.uplink_rate}s < step time {time}s')
            elif u.is_free:
                pass
            else:
                raise RuntimeError('unknown user state')

    def stationary_time(self):
        # 更新信道内传输速率
        self.update_uplink_rate()
        # 最小时间为时隙时间
        min_time = self.slot_time
        for u in self.UEs:
            if u.is_inferring:
                if(u.head_time_left > 0):
                    time = u.head_time_left
                elif(u.tail_time_left > 0):
                    time = u.tail_time_left
                else:
                    raise RuntimeError('user has no inferring time left')
                # if(u.in_local_mode()):
                #     time = u.head_time_left
                # elif(u.in_mec_mode()):
                #     if(u.data_left > 0):
                #         time = u.head_time_left
                #     else:
                #         time = u.tail_time_left
                # time = u.time_left
            elif u.is_offloading:
                time = u.data_left / u.uplink_rate  # todo: divide by zero?
            elif u.is_free:
                time = self.slot_time
            else:
                raise RuntimeError('unknown user state')
            if time < min_time:
                min_time = time
            # 返回的肯定比时隙小
        return min_time

    def update_uplink_rate(self):
        for channel in self.channels:
            channel.reset()
        for u in self.UEs:
            if u.is_offloading:
                channel_index = u.channel
                self.channels[channel_index].new_occupation(u)

        for channel in self.channels:
            channel.update_uplink_rate()

    def assign_action(self, action):
        # action是10个终端的划分点
        for u, a in zip(self.UEs, action):
            point = a[0]
            channel = a[1]
            power = a[2]
            u.next_point = point
            u.next_channel = channel
            u.power = power

    def is_done(self):
        for u in self.UEs:
            if not u.is_free:
                return False
        return True
