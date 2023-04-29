from finrl.agents.stablebaselines3.models import PPO, A2C, DDPG


class RLModels:
    def __init__(self, model_name, env):
        self.model_name = model_name
        self.env = env
        self.model = self.get_model(model_name, env)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, data):
        return self.model.predict(data)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = self.model.load(path)

    @staticmethod
    def get_model(model_name, env):
        if model_name == "ppo":
            model = PPO('MlpPolicy', env)
        elif model_name == "a2c":
            model = A2C('MlpPolicy', env)
        elif model_name == "ddpg":
            model = DDPG('MlpPolicy', env)
            
        
        else:
            raise ValueError("Invalid model_name")
        return model

