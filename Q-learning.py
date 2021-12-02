import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import gym


class Q_learning():
    def __init__(self, opt):
        self.epochs = opt.epochs
        self.ep_list = []
        self.reward_list = []
        self.frames = []
        
    def train(self):
        env = gym.make("MountainCar-v0")
        q_table_size, q_table = self.init_qtable(env)
        episode_rewards= []
        end_point = env.goal_position
        
        
        for ep in range(1, self.epochs+1):
            ep_reward = 0 #initailize
            state = self.define_state(env.reset(), env)
            done = False
            while not done:
                action = np.argmax(q_table[state])
                new_state, reward, done, _= env.step(action)
                ep_reward += reward
                update_state = self.define_state(new_state, env)
                
                if ep % 10000 == 0:
                    self.frames.append(env.render(mode="rgb_array"))
                    env.render()
                    time.sleep(0.02)
                    
                if not done:
                    max_q_value = np.max(q_table[update_state])
                    curr_q_value = q_table[state + (action,)]
                    new_q_value = (1- opt.lr) * curr_q_value + opt.lr * (reward + opt.factor * max_q_value)
                    q_table[state + (action,)] = new_q_value 
                elif new_state[0] >= end_point:
                    q_table[state + (action,)] = 0
                state = update_state   
                
                
            episode_rewards.append(ep_reward)
            if not ep % 50 :
                reward_sum = sum(episode_rewards[-50:]) / 50
                self.ep_list.append(ep)
                self.reward_list.append(reward_sum)
                print(f'{ep} Episode || Reward : {reward_sum}')
        
        self.save_plot(self.ep_list, self.reward_list)
        self.save_frames_as_gif(self.frames)
        env.close()
    
    def init_qtable(self , env):
        qtable_size = [opt.env_size] * len(env.observation_space.high)
        q_table = np.random.uniform(low=0,high=10, size = (qtable_size + [env.action_space.n]))
        return qtable_size, q_table
    
    def define_state(self, state, env):
        scale = ((env.observation_space.high - env.observation_space.low) / ([opt.env_size, opt.env_size]))
        state = (state - env.observation_space.low) / scale
        output = tuple(state.astype(np.int32))
        return output
        
    def save_plot(self, ep, reward):
        save_root_name = "./result.png"
        fig = plt.figure(figsize=(12, 8))
        plt.plot(ep, reward)
        plt.xlabel('episode')
        plt.ylabel('Reward')
        plt.savefig(save_root_name)
        
    def save_frames_as_gif(self, frames, path='./', filename='gym_animation.gif'):
        plt.figure(figsize=(frames[0].shape[1] / 100.0, frames[0].shape[0] / 100.0), dpi=50)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
        anim.save(path + filename, writer='Pillow', fps=60)
        print("File saved!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default = 20000, help = "train Epochs")
    parser.add_argument("--env_size", type=int, default=40, help="size of environment")
    parser.add_argument("--factor", type=float, default = 0.95, help="Discount factor")
    parser.add_argument("--lr", type=float, default = 0.1, help="learning rate")
    
    opt = parser.parse_args()
    
    env = gym.make("MountainCar-v0")
    q_learning = Q_learning(opt)
    q_learning.train()