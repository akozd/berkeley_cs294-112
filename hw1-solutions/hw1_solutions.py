import gym
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, LeakyReLU, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
import load_policy
import pickle


def load_behavior_cloning_policy(data):
    
    obs_train, _, act_train, _, input_dim, output_dim, _ , _ = data

    model = keras.Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='linear'))
    model.add(LeakyReLU(0.01))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(),loss='mse', metrics=['mse'])
    model.fit(obs_train, act_train, epochs=10, batch_size=32)
    
    return model


def load_poor_behavior_cloning_policy(data):
    
    obs_train, _, act_train, _, input_dim, output_dim, _ , _ = data

    model = keras.Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='linear'))
    model.add(LeakyReLU(0.01))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='linear'))
    model.add(LeakyReLU(0.01))
    model.add(BatchNormalization())
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(),loss='mse', metrics=['mse'])
    model.fit(obs_train, act_train, epochs=10, batch_size=32)
    
    return model

def load_hyperparameter_experimentation_cloning_policy(data, stop):
    
    obs_train, _, act_train, _, input_dim, output_dim, _ , _ = data
    obs_train = obs_train[:stop]
    act_train = act_train[:stop]

    model = keras.Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='linear'))
    model.add(LeakyReLU(0.01))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(),loss='mse', metrics=['mse'])
    model.fit(obs_train, act_train, epochs=10, batch_size=32, verbose=False)
    
    return model


    
def load_DAgger(data):

    obs_train, _, act_train, _, input_dim, output_dim, = data

    model = keras.Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='linear'))
    model.add(LeakyReLU(0.01))
    model.add(Dense(128, input_dim=input_dim, activation='linear'))
    model.add(LeakyReLU(0.01))
    model.add(Dense(output_dim, activation='linear'))
    model.compile(optimizer=keras.optimizers.Adam(),loss='mse', metrics=['mse'])
    model.fit(obs_train, act_train, epochs=20, batch_size=32)

    return model


def load_data(args):

    with open(args.expert_data, 'rb') as f:
        data = pickle.load(f)
        obs = data['observations']
        act = data['actions']
    
    obs_train, obs_test, act_train, act_test = train_test_split(obs, act, test_size=0.1)
    act_train = act_train.reshape(act_train.shape[0], act_train.shape[2])
    act_test = act_test.reshape(act_test.shape[0], act_test.shape[2])
    
    mean = np.mean(obs_train, axis=0)
    std = np.std(obs_train, axis=0)
    obs_train = (obs_train-mean)/(std+1e-6)
    obs_test = (obs_test-mean)/(std+1e-6)

    input_dim = obs.shape[1] 
    output_dim = act.shape[2]

    return obs_train, obs_test, act_train, act_test, input_dim, output_dim, mean, std


def policy_rollout(data, envname, policy_fn, num_rollouts, max_timesteps, render):

    env = gym.make(envname)
    _, _, _, _, _, _, mean, std = data
    rewards = []
    observations = []
    actions = []
    best_reward = 0.0
    best_model = None
    
    for _ in range(num_rollouts):
        total_reward = 0.0
        step = 0
        done = False
        observation = env.reset()
        
        while not done:
            norm_obs = (observation - mean)/(std+1e-6)
            action = policy_fn.predict(np.array([norm_obs]))
            observations.append(observation)
            actions.append(action)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1
            if render: env.render()
            if done: break
            if step >= max_timesteps: break
        rewards.append(total_reward)
        if total_reward > best_reward: 
            best_reward = total_reward
            best_model = policy_fn
    
    return rewards, observations, actions, best_reward, best_model
    

def problem_hw1_2_2_1(args):

    data = load_data(args)
    policy_fn = load_behavior_cloning_policy(data)
    
    rewards, _, _, _, _ = policy_rollout(
        data,
        args.envname,
        policy_fn, 
        args.num_rollouts, 
        args.max_timesteps, 
        args.render)

    print('returns ', rewards)
    print('mean return ', np.mean(rewards))
    print('std of return ', np.std(rewards))

def problem_hw1_2_2_2(args):

    data = load_data(args)
    policy_fn = load_poor_behavior_cloning_policy(data)

    rewards, _, _, _, _ = policy_rollout(
        data,
        args.envname,
        policy_fn, 
        args.num_rollouts, 
        args.max_timesteps, 
        args.render)

    print('rewards ', rewards)
    print('mean reward ', np.mean(rewards))
    print('std of reward ', np.std(rewards))


def problem_hw1_2_3(args):

    # we want number of demonstrations
    data = load_data(args)
    for i in range(1800,19800,1800):
        policy_fn = load_hyperparameter_experimentation_cloning_policy(data, i)
        rewards, _, _, _, _ = policy_rollout(
            data,
            args.envname,
            policy_fn, 
            args.num_rollouts, 
            args.max_timesteps, 
            args.render)
        print('trained on {} observations: '.format(i))
        print('mean reward ', np.mean(rewards))
        print('std of reward ', np.std(rewards))

def problem_hw1_3(args):

    # DAgger:
    # 1) train on data
    # 2) perform rollouts
    # 3) get expert policy to label observations
    # 4) merge datasets
    # 5) go to step 1

    data = load_data(args)
    obs_train, obs_test, act_train, act_test, input_dim, output_dim, _, _ = data
    expert_policy = load_policy.load_policy(args.expert_policy_file)    
    mean_DAgger = []
    stddev_DAgger = []

    for _ in range(args.num_DAgger_iterations):

        policy_fn = load_DAgger([obs_train, obs_test, act_train, act_test, input_dim, output_dim])
    
        rewards, observations, _, _, _ = policy_rollout(
            data,
            args.envname,
            policy_fn, 
            args.num_rollouts, 
            args.max_timesteps, 
            args.render)
        mean_DAgger.append(np.mean(rewards))
        stddev_DAgger.append(np.std(rewards))

        expert_actions = []
        with tf.Session() as sess:
            for obs in observations:
                expert_action = expert_policy(np.array([obs]))
                expert_actions.append(expert_action)     
        expert_actions = np.array(expert_actions)
        expert_actions = np.reshape(expert_actions, [expert_actions.shape[0], expert_actions.shape[2]])
        obs_train = np.concatenate([obs_train, observations], axis=0)
        act_train = np.concatenate([act_train, expert_actions], axis=0)

    rewards, _, _, _, _ = policy_rollout(
        data,
        args.envname,
        policy_fn, 
        args.num_rollouts, 
        args.max_timesteps, 
        args.render)

    print('mean returns per DAgger iteration: ', mean_DAgger)
    print('standard deviation of rewards per DAgger iteration: ', stddev_DAgger)
    print('returns generated by rollout after last DAgger iteration: ', rewards)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('problem', type=str)
    parser.add_argument('expert_data', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--expert_policy_file', type=str)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--max_timesteps', type=int, default=10000000)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_DAgger_iterations', type=int, default=10)
    args = parser.parse_args()

    if args.problem == 'hw1.2.2.1':
        problem_hw1_2_2_1(args)
    elif args.problem == 'hw1.2.2.2':
        problem_hw1_2_2_2(args)
    elif args.problem == 'hw1.2.3':
        problem_hw1_2_3(args)
    elif args.problem == 'hw1.3':
        problem_hw1_3(args)
    else:
        raise ValueError('The problem "{}" is not a valid option'.format(args.problem))   

if __name__ == '__main__':
    main()  