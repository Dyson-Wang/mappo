import argparse

from agent import HPPO
from env import MECsystem
from utils import *


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
    200 * 0.15 / 0.5
    max_episode_step = int((args.possion_lambda * 0.15) / args.slot_time)
    while True:
        s_t = env.reset()
        for j in range(max_episode_step):
            actions = agent.select_action(s_t)
            s_t1, r_t, done, _ = env.step(actions)

            agent.buffer.states.append(s_t)
            agent.buffer.rewards.append(r_t)
            agent.buffer.is_terminals.append(done)

            global_step += 1
            s_t = s_t1

            if global_step % args.step == 0:
                agent.update()
                test(args, global_episode, global_step, test_env, agent, logger)

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

    parser.add_argument('--max_global_step', type=int, default=50000)
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
