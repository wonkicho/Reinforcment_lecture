from bandit import BanditTwoArmedHighLowFixed
import numpy as np
env = BanditTwoArmedHighLowFixed()

count = np.zeros(2)
sum_rewards = np.zeros(2)
Q = np.zeros(2)
alpha = np.ones(2)
beta = np.ones(2)

def thompson_sampling(alpha, beta):
    samples = [np.random.beta(alpha[i]+1, beta[i]+1) for i in range(2)]
    return np.argmax(samples)

num_rounds = 100
for i in range(num_rounds):
    arm = thompson_sampling(alpha,beta)    
    next_state, reward, done, info = env.step(arm)
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]
    if reward == 1:
        alpha[arm] = alpha[arm] + 1
    else:
        beta[arm] = beta[arm] + 1

print(Q)
print('The optimal arm is arm {}'.format(np.argmax(Q)+1))

