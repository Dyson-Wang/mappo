import argparse

from agent import HPPO
from env import MECsystem
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


# 清空日志文件和绘图文件夹
log_file = "training_log_metrics_weight.txt"
plot_dir = "plots_metrics_weight"

if os.path.exists(log_file):
    open(log_file, "w").close()

if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)
os.makedirs(plot_dir)

# 日志记录函数
def log_message(message):
    with open(log_file, "a") as f:
        f.write(message + "\n")

def plot_all_metrics(metrics_dict, episode):
    """
    将所有指标绘制到一个包含多个子图的图表中
    - 对曲线进行平滑处理
    - 添加误差带显示
    
    参数:
    metrics_dict: 包含所有指标数据的字典，格式为 {metric_name: values_list}
    episode: 当前的episode数
    """
    # 创建一个2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics (Up to Episode {episode})', fontsize=16)
    
    # 压平axes数组以便迭代
    axes = axes.flatten()
    
    # 为每个指标获取x轴值
    any_metric = list(metrics_dict.values())[0]
    x_values = [50 * (i + 1) for i in range(len(any_metric))]
    
    # 平滑参数 - 窗口大小
    window_size = min(5, len(x_values)) if len(x_values) > 0 else 1
    
    # 在每个子图中绘制一个指标
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        if i >= 5:  # 我们只有5个指标
            break
            
        ax = axes[i]
        values_array = np.array(values)
        
        # 应用平滑处理
        if len(values) > window_size:
            # 创建平滑曲线
            smoothed = np.convolve(values_array, np.ones(window_size)/window_size, mode='valid')
            
            # 计算滚动标准差用于误差带
            std_values = []
            for j in range(len(values) - window_size + 1):
                std_values.append(np.std(values_array[j:j+window_size]))
            std_values = np.array(std_values)
            
            # 调整x轴以匹配平滑后的数据长度
            smoothed_x = x_values[window_size-1:]
            
            # 绘制平滑曲线和原始散点
            ax.plot(smoothed_x, smoothed, '-', linewidth=2, label='Smoothed')
            ax.scatter(x_values, values, alpha=0.3, label='Original')
            
            # 添加误差带
            ax.fill_between(smoothed_x, smoothed-std_values, smoothed+std_values, 
                           alpha=0.2, label='±1 StdDev')
        else:
            # 如果数据点太少，只绘制原始数据
            ax.plot(x_values, values, 'o-', label='Data')
        
        ax.set_title(metric_name.replace('_', ' '))
        ax.set_xlabel('Episodes')
        ax.set_ylabel(metric_name.replace('_', ' '))
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 删除未使用的子图
    if len(metrics_dict) < 6:
        fig.delaxes(axes[5])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plot_dir, f'training_metrics.png'))
    plt.close(fig)


def train(args, agent, env, test_env, save_dir, logger, start_episode=0, start_step=0):
    global_episode = start_episode
    global_step = start_step
    # for i in range(50):
    #     actions = [
    #         (i, 0, 2.8490683436393738),
    #         (i, 1, 2.9760618209838867), 
    #         (i, 2, 2.4263856410980225), 
    #         (i, 3, 2.5711382031440735), 
    #         (i, 4, 2.6409727334976196), 
    #         (i, 5, 2.3382599353790283), 
    #         (i, 6, 2.112567901611328), 
    #         (i, 7, 2.5569998025894165), 
    #         (i, 8, 2.5565664768218994), 
    #         (i, 9, 2.5968937873840332)
    #         ]
    #     print(f'episode 1 actions {actions}')
    #     s_t = env.reset()
    #     done = False
    #     test_reward = 0
    #     time_used = 0
    #     energy_used = 0
    #     finished = 0
    #     while True:
    #         s_t1, r_t, done, info = env.step(actions)
    #         test_reward += r_t
    #         time_used += info['total_time_used']
    #         energy_used += info['total_energy_used']
    #         finished += info['total_finished']
    #         if done:
    #             avg_time_used = time_used / finished
    #             avg_energy_used = energy_used / finished
    #             logger.info(f'step {i}, reward {test_reward:.4f}, ({avg_time_used:.6f}s, {time_used:.6f}s, {avg_energy_used:.6f}j, {energy_used:.6f}j,)/task/device, {finished} tasks finished')
    #             break
    # return
    # 200 * 0.15 / 0.5 = 600
    max_episode_step = int((args.possion_lambda * 0.15) / args.slot_time)

    total_rewards_per_episode = []
    episode_lengths = []
    policy_losses = []
    value_losses = []
    entropies = []

    # 每50个episode的平均值列表
    avg_total_rewards_per_50 = []
    avg_episode_length_per_50 = []
    avg_policy_loss_per_50 = []
    avg_value_loss_per_50 = []
    avg_entropy_per_50 = []
    while True:
        s_t = env.reset()
        episode_reward = 0.0
        steps = 0
        for j in range(max_episode_step):
            steps += 1
            actions = agent.select_action(s_t)
            s_t1, r_t, done, _ = env.step(actions)

            step_reward = np.mean(r_t)
            episode_reward += step_reward

            agent.buffer.states.append(s_t)
            agent.buffer.rewards.append(r_t)
            agent.buffer.is_terminals.append(done)

            global_step += 1
            s_t = s_t1

            if global_step % args.step == 0:
                a_loss, c_loss, ent = agent.update()
                test(args, global_episode, global_step, test_env, agent, logger)
                total_rewards_per_episode.append(episode_reward)
                episode_lengths.append(steps)
                policy_losses.append(a_loss)
                value_losses.append(c_loss)
                entropies.append(ent)

                avg_reward_50 = np.mean(total_rewards_per_episode[-50:])
                avg_length_50 = np.mean(episode_lengths[-50:])
                avg_policy_loss_50 = np.mean(policy_losses[-50:])
                avg_value_loss_50 = np.mean(value_losses[-50:])
                avg_entropy_50 = np.mean(entropies[-50:])

                avg_total_rewards_per_50.append(avg_reward_50)
                avg_episode_length_per_50.append(avg_length_50)
                avg_policy_loss_per_50.append(avg_policy_loss_50)
                avg_value_loss_per_50.append(avg_value_loss_50)
                avg_entropy_per_50.append(avg_entropy_50)

                log_message(f"Episode {global_episode}: "
                            f"AvgTotalReward(last50)={avg_reward_50:.3f}, "
                            f"AvgEpisodeLength(last50)={avg_length_50:.3f}, "
                            f"AvgPolicyLoss(last50)={avg_policy_loss_50:.3f}, "
                            f"AvgValueLoss(last50)={avg_value_loss_50:.3f}, "
                            f"AvgEntropy(last50)={avg_entropy_50:.3f}")
                    
                # 创建指标字典
                metrics_dict = {
                    "Average_Total_Reward": avg_total_rewards_per_50,
                    "Average_Episode_Length": avg_episode_length_per_50,
                    "Average_Policy_Loss": avg_policy_loss_per_50,
                    "Average_Value_Loss": avg_value_loss_per_50, 
                    "Average_Entropy": avg_entropy_per_50
                }
                    
                    # 调用新的绘图函数
                plot_all_metrics(metrics_dict, global_step)
            if done:
                global_episode += 1
                break
        
        if global_step > args.max_global_step:
            # test(args, global_episode, global_step, test_env, agent, logger)
            break

    agent.save_model(save_dir + 'ckp.pt', args)


def test(args, episode, step, test_env, agent, logger=None):
    done = False
    s_t = test_env.reset()
    test_reward = 0
    time_used = 0
    energy_used = 0
    finished = 0
    split_point_avg = 0
    avg_channel = 0
    avg_power = 0
    while not done:
        actions = agent.select_action(s_t, test=True)
        s_t1, r_t, done, info = test_env.step(actions)
        s_t = s_t1
        test_reward += sum(r_t) / len(r_t)
        time_used += info['total_time_used']
        energy_used += info['total_energy_used']
        finished += info['total_finished']
        points = [actions[0] for actions in actions]
        channels = [actions[1] for actions in actions]
        powers = [actions[2] for actions in actions]
        avg_point = sum(points) / len(points)
        avg_channel = sum(channels) / len(channels)
        avg_power = sum(powers) / len(powers)
        if(split_point_avg == 0):
            split_point_avg = avg_point

        else:
            split_point_avg += avg_point
            split_point_avg /= 2
        if(avg_channel == 0):
            avg_channel = avg_channel
        else:
            avg_channel += avg_channel
            avg_channel /= 2
        if(avg_power == 0):
            avg_power = avg_power
        else:
            avg_power += avg_power
            avg_power /= 2
    # print(finished)

    avg_time_used = time_used / finished
    avg_energy_used = energy_used / finished
    if logger is not None:
        logger.info(f'step {step}, reward {test_reward:.4f}, point_avg {split_point_avg}, ({avg_time_used:.6f}s, {avg_energy_used:.6f}j,)/task/device, avg_channel {avg_channel}, avg_power {avg_power}')


def init_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='default_DRL')

    # system
    parser.add_argument('--net', default='resnet152', type=str, help='可选：resnet18, resnet152, mobilenetv2, vgg11')
    parser.add_argument('--possion_lambda', default=200, type=int)
    parser.add_argument('--num_channels', default=5, type=int)
    parser.add_argument('--num_users', default=10, type=int)
    parser.add_argument('--num_user_state', default=4, type=int)
    parser.add_argument('--pmax', default=3, type=float, help='max power')
    parser.add_argument('--dmax', default=100, type=float, help='max distance')
    parser.add_argument('--dmin', default=1, type=float, help='min distance')
    parser.add_argument('--beta', default=0.5, type=float)

    # channel
    parser.add_argument('--path_loss_exponent', default=3, type=float)
    parser.add_argument('--width', default=1e7, type=float)
    parser.add_argument('--noise', default=1e-9, type=float)

    # PPO
    parser.add_argument('--lr_a', default=0.0005, type=float, help='actor net learning rate')
    parser.add_argument('--lr_c', default=0.001, type=float, help='critic net learning rate')

    parser.add_argument('--max_global_step', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--slot_time', default=0.6, type=float)

    parser.add_argument('--repeat_time', default=20, type=int)
    parser.add_argument('--step', default=500, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lam', default=0.97, type=float)
    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--w_entropy', default=0.05, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_parser()
    exp_name = f'{args.net}_MAHPPO'
    os.makedirs(os.path.join('result', exp_name), exist_ok=True)
    logger = setup_logger(__name__, os.path.join('result', exp_name))

    d_args = vars(args)
    for k in d_args.keys():
        logger.info(f'{k}: {d_args[k]}')

    user_params = {
        'num_channels': args.num_channels,
        'num_points': 50,
        'possion_lambda': args.possion_lambda,
        'pmax': args.pmax,
        'dmin': args.dmin,
        'dmax': args.dmax,
        'net': args.net,
        'test': False
    }
    test_user_params = {
        'num_channels': args.num_channels,
        'num_points': 50,
        'possion_lambda': args.possion_lambda,
        'pmax': args.pmax,
        'dmin': args.dmin,
        'dmax': args.dmax,
        'net': args.net,
        'test': True
    }
    channel_params = {
        'path_loss_exponent': args.path_loss_exponent,
        'width': args.width,
        'noise': args.noise
    }
    agent_params = {
        'num_users': args.num_users,
        'num_states': args.num_users * args.num_user_state,
        'num_channels': args.num_channels,
        'num_splits': 49,
        'lr_a': args.lr_a,
        'lr_c': args.lr_c,
        'pmax': args.pmax,
        'gamma': args.gamma,
        'lam': args.lam,
        'repeat_time': args.repeat_time,
        'batch_size': args.batch_size,
        'eps_clip': args.eps_clip,
        'w_entropy': args.w_entropy,
    }

    env = MECsystem(args.slot_time, args.num_users, args.num_channels, user_params, channel_params, args.beta)
    test_env = MECsystem(args.slot_time, args.num_users, args.num_channels, test_user_params, channel_params, args.beta)
    agent = HPPO(**agent_params)

    train(args, agent, env, test_env, os.path.join('result', exp_name), logger)
