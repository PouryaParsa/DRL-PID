from line_follower import LineFollower
import numpy as np
import rospy
from summit_description.srv import StartUp, StartUpRequest  # service call
from sac_torch import Agent
from utils import plot_learning_curve
from utils import str2bool
import torch as T
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='track_car')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    # ... (add other arguments)

    return parser.parse_args()


def setup_rospy():
    rospy.wait_for_service('/Activate')
    service_call = rospy.ServiceProxy('/Activate', StartUp)
    response = service_call(True)
    print(response)


def create_agent(args):
    return Agent(alpha=args.lr, beta=args.lr, n_actions=args.action, gamma=args.gamma,
                 reward_scale=args.reward_scale, layer1_size=args.hidden_size,
                 layer2_size=args.hidden_size, tau=args.tau)


def run_rl_agent(agent, env, args):
    env.reset()
    observation, _, done = env.feedback()
    done = False
    score = 0
    step = 0

    while not done:
        time_step = 0
        action = agent.choose_action(observation, warmup=args.warmup,
                                      evaluate=args.load, action=args.action)

        if args.action == 6:
            env.update_pid(kp=(action[0] + 1) * 2, kd=(action[1] + 1) * 2,
                            kp2=(action[2] + 1) * 2, kd2=(action[3] + 1) * 2,
                            ki=(action[4] + 1) / 2, ki2=(action[5] + 1) / 2)
        else:
            env.update_pid(kp=(action[0] + 1) * 2, kd=(action[1] + 1) * 2,
                           kp2=(action[2] + 1) * 2, kd2=(action[3] + 1) * 2)

        while time_step < 3:
            done = env.done
            if done:
                break
            env.control(sac_pid=args.RL)
            env.rate.sleep()
            time_step += 1

        step += 1
        observation_, reward, done = env.feedback()
        score += reward
        observation = observation_

    return score


def save_error_data(env, args):
    filename = 'data/test/test_RL-PID.txt' if args.RL else 'data/test/test_PID.txt'
    with open(filename, 'a') as file_object:
        file_object.write(str(env.errorx_list) + '\n')


def main():
    args = parse_arguments()
    setup_rospy()
    agent = create_agent(args)
    env = LineFollower()
    episode = args.episode
    score_history = []
    score_save = []

    def shutdown():
        print("shutdown!")
        env.stop()
        service_call(False)

    rospy.on_shutdown(shutdown)

    mean_error = []
    agent.load_models()
    agent.save_models()
    print(T.cuda.is_available())

    score = run_rl_agent(agent, env, args)

    score_save.append(score)
    avg_score = np.mean(score_history[-50:])

    print('score = %.2f' % score, 'step:', step)

    if args.RL:
        save_error_data(env, args)


if __name__ == '__main__':
    main()





