import gym
from stable_baselines3 import a2c

env = gym.make('Blackjack-v1')

obs = env.reset()

model = a2c.A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000)

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    terminated = False
    while not terminated:
        # env.render()
        obs, reward, terminated, info = env.step(model.predict(obs)[0])
        print(reward)
    

env.close()



