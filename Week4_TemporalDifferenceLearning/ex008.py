import numpy as np
import gym
env = gym.make('FrozenLake-v1')
from tqdm import tqdm

def action_epsilon_greedy(q, s, epsilon=0.05):
    if np.random.rand() > epsilon:
        return np.argmax(q[s])
    return np.random.randint(4)

def evaluate_policy(q, n=500):
    acc_returns = 0
    for i in range(n):
        done = False
        s = env.reset()
        while not done:
            a = action_epsilon_greedy(q, s, epsilon=0.)
            s, reward, done, _ = env.step(a)
            acc_returns += reward
    return acc_returns / n

def sarsa(alpha=0.02, gamma=1., epsilon=0.05, q=None, env=env):
    
    if q is None:
        q = np.zeros((16,4)).astype(np.float32)

    nb_episodes = 200000
    steps = 2000
    progress = []
    for i in tqdm(range(nb_episodes)):
        done = False
        s = env.reset()
        a = action_epsilon_greedy(q, s, epsilon=epsilon)
        while not done:
            new_s, reward, done, _ = env.step(a)
            new_a = action_epsilon_greedy(q, new_s, epsilon=epsilon)
            q[s,a] = q[s,a] + alpha * (reward + gamma * q[new_s,new_a] - q[s,a])
            s = new_s
            a = new_a

        if i%steps == 0:
            progress.append(evaluate_policy(q, n=500))

    return q, progress

q, progress = sarsa(alpha=0.02, epsilon=0.05, gamma=0.999)
print(evaluate_policy(q, n=10000))
print(progress)


