import gymnasium as gym
import numpy as np



class SmartClimateControlEnv(gym.Env):
    def __init__(self, time_limit:int=100, options:dict=None):
        super(SmartClimateControlEnv, self).__init__()

        self.time_limit = time_limit
        self.time = 0
        
        if options is None:
            options = {}
        
        # Targets
        if options.get('random_target', False):
            self.target_temp = self.np_random.uniform(low=0.0, high=100.0)
            self.target_hum = self.np_random.uniform(low=0.0, high=100.0)
        else:
            if options.get("target_temp") is not None:
                self.target_temp = options["target_temp"]
            else:
                self.target_temp = self.np_random.uniform(low=0.0, high=100.0)

            if options.get("target_humidity") is not None:
                self.target_hum = options["target_humidity"]
            else:
                self.target_hum = self.np_random.uniform(low=0.0, high=100.0)
        
        # State variables
        self.temperature = self.np_random.uniform(low=self.target_temp - 1, high=self.target_temp + 1)  # Celsius
        self.humidity = self.np_random.uniform(low=self.target_hum - 3, high=self.target_hum + 3) # Percentage

        # Change factors (per unit/second at 100% power)
        # Heater
        self.heater_temp_rate = 2.0 / 60.0  # 2 degrees in 60 seconds => 0.0333 per second
        # Humidifier
        self.humidifier_hum_rate = 5.0 / 60.0 # 5% in 60 seconds => 0.0833 per second
        # Fan (temperature and humidity change)
        self.fan_temp_rate = -1.5 / 60.0    # -1.5 in 60 seconds => -0.025 per second
        self.fan_hum_rate = -3.0 / 60.0     # -3% in 60 seconds => -0.05 per second

        # Power consumption (watts)
        self.heater_power_consumption = 150.0
        self.humidifier_power_consumption = 10.0
        self.fan_power_consumption = 25.0

        # Action space: [heater_power, humidifier_power, fan_power]
        # Each one from 0 to 100 percent
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: [temperature, humidity, target_temperature, target_humidity]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), # minimum temperature and humidity
            high=np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32), # maximum temperature and humidity
            dtype=np.float32
        )

    def reset(self, seed=None, options:dict=None):
        super().reset(seed=seed)
        self.time = 0
        
        if options is None:
            options = {}
        
        # Targets
        if options.get('random_target', False):
            self.target_temp = self.np_random.uniform(low=0.0, high=100.0)
            self.target_hum = self.np_random.uniform(low=0.0, high=100.0)
        else:
            if options.get("target_temp") is not None:
                self.target_temp = options["target_temp"]
            else:
                self.target_temp = self.np_random.uniform(low=0.0, high=100.0)

            if options.get("target_humidity") is not None:
                self.target_hum = options["target_humidity"]
            else:
                self.target_hum = self.np_random.uniform(low=0.0, high=100.0)
            
    
        # Reset to initial state near target
        self.temperature = self.np_random.uniform(low=self.target_temp - 1, high=self.target_temp + 1)
        self.humidity = self.np_random.uniform(low=self.target_hum - 3, high=self.target_hum + 3)
        
        # Clamp values
        self.target_temp = np.clip(self.target_temp, 0.0, 100.0)
        self.target_hum = np.clip(self.target_hum, 0.0, 100.0)
        self.temperature = np.clip(self.temperature, 0.0, 100.0)
        self.humidity = np.clip(self.humidity, 0.0, 100.0)
        
        
        # Initialize previous errors
        self.prev_temp_error = abs(self.temperature - self.target_temp)
        self.prev_hum_error = abs(self.humidity - self.target_hum)

        # Initialize previous action
        self.prev_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        observation = np.array([self.temperature, self.humidity, self.target_temp, self.target_hum], dtype=np.float32)
        return observation, {}
        
    def step(self, action):
        # action = [heater_power, humidifier_power, fan_power]
        heater_pwr, humidifier_pwr, fan_pwr = action

        # Calculate changes in this step
        temp_change = 0.0
        hum_change = 0.0
        total_power = 0.0

        # Changes due to heating
        temp_change += heater_pwr * self.heater_temp_rate
        total_power += heater_pwr * self.heater_power_consumption

        # Changes due to humidifier
        hum_change += humidifier_pwr * self.humidifier_hum_rate
        total_power += humidifier_pwr * self.humidifier_power_consumption

        # Changes due to fan
        temp_change += fan_pwr * self.fan_temp_rate
        hum_change += fan_pwr * self.fan_hum_rate
        total_power += fan_pwr * self.fan_power_consumption

        # Update state
        self.temperature += temp_change
        self.humidity += hum_change

        # Clamp values
        self.temperature = np.clip(self.temperature, 0.0, 100.0)
        self.humidity = np.clip(self.humidity, 0.0, 100.0)

        # Calculate reward 
        reward, temp_error, hum_error,\
        energy_weight, energy_norm, error_norm = self.calculate_reward(total_power, action)
        
        observation = np.array([self.temperature, self.humidity, self.target_temp, self.target_hum], dtype=np.float32)
        self.time += 1
        terminated = self.time >= self.time_limit
        truncated = False
        
        
        info = {
            "temp_error": temp_error,
            "hum_error": hum_error,
            "energy_weight": energy_weight,
            "energy_norm": energy_norm,
            "error_norm": error_norm,
        }
        
        
        info = {"power_consumption": total_power, "temp_error": temp_error, "hum_error": hum_error}

        return observation, reward, terminated, truncated, info


    def calculate_reward(self, total_power, action):
        temp_error = abs(self.temperature - self.target_temp)
        hum_error = abs(self.humidity - self.target_hum)

        # normalization
        energy_norm = total_power / 185.0
        error_norm = (temp_error + hum_error) / 200.0
        action_change_norm = np.sum(np.abs(action - self.prev_action))

        # energy weight increases as error decreases (bounded)
        energy_weight = 2.0 + 2.0 * (1.0 - error_norm)  # ranges [2,4]
        # or piecewise if you prefer sharper effect

        reward = - energy_weight * energy_norm - 1.0 * error_norm - 0.05 * action_change_norm


        # update prevs
        self.prev_temp_error = temp_error
        self.prev_hum_error = hum_error
        self.prev_action = action.copy()

        return reward, temp_error, hum_error, energy_weight, energy_norm, error_norm

    def render(self):
        print(f"Temp: {self.temperature:.2f}, Humidity: {self.humidity:.2f}")