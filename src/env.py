import gymnasium as gym
import numpy as np

class SmartClimateControlEnv(gym.Env):
    def __init__(self, time_limit:int=100, options:dict=None, rate_scale_range=(1.0, 1.0)):
        super(SmartClimateControlEnv, self).__init__()

        self.rate_scale_range = rate_scale_range
        self.time_limit = time_limit
        self.time = 0

        # Initialize noise settings
        self.DEFAULT_NOISE = {
            'sensor_noise': {'enabled': False, 'std': 0.1},
            'env_disturbance': {'enabled': False, 'std': 0.05},
            'power_fluctuation': {'enabled': False, 'std': 0.05},
            'adaptive_noise': {'enabled': False, 'base_std': 0.05, 'scale_factor': 0.1}
        }
        
        self.noise_config = self._merge_noise_config(options.get('noise_config') if options else None)
           
        # Initialize for adaptive noise
        self.avg_error = 0.0
        self.error_history = []
        
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
        self.temperature = self.np_random.uniform(low=self.target_temp - 1, high=self.target_temp + 1)
        self.humidity = self.np_random.uniform(low=self.target_hum - 3, high=self.target_hum + 3)

        # Base rates and power
        self.base_heater_temp_rate = 2.0 / 30.0
        self.base_humidifier_hum_rate = 5.0 / 30.0
        self.base_fan_temp_rate = -1.5 / 30.0
        self.base_fan_hum_rate = -3.0 / 30.0

        self.base_heater_power_consumption = 150.0
        self.base_humidifier_power_consumption = 10.0
        self.base_fan_power_consumption = 25.0

        self.apply_rate_scale()

        # Action space: [heater_power, humidifier_power, fan_power]
        # Each one from 0 to 100 percent
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: [temperature, humidity, target_temperature, target_humidity]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )

    def apply_rate_scale(self):
        scale = self.np_random.uniform(*self.rate_scale_range)
        
        self.heater_temp_rate    = self.base_heater_temp_rate * scale
        self.humidifier_hum_rate = self.base_humidifier_hum_rate * scale
        self.fan_temp_rate       = self.base_fan_temp_rate * scale
        self.fan_hum_rate        = self.base_fan_hum_rate * scale

        self.heater_power_consumption     = self.base_heater_power_consumption * scale
        self.humidifier_power_consumption = self.base_humidifier_power_consumption * scale
        self.fan_power_consumption        = self.base_fan_power_consumption * scale

    def reset(self, seed=None, options:dict=None):
        super().reset(seed=seed)

        self.apply_rate_scale()
        self.time = 0

        self.noise_config = self._merge_noise_config(options.get('noise_config') if options else None)
        
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
        self.prev_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        observation = np.array([self.temperature, self.humidity, self.target_temp, self.target_hum], dtype=np.float32)
        return observation, {}

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)
        
        # Apply power fluctuation
        fluctuation_enabled = self.noise_config['power_fluctuation']['enabled']
        fluctuation_std = self.noise_config['power_fluctuation'].get('std', 0.05)
        if fluctuation_enabled:
            action = np.clip(action + self.np_random.normal(0, fluctuation_std, size=action.shape), 0.0, 1.0)

        heater_pwr, humidifier_pwr, fan_pwr = action

        temp_change = 0.0
        hum_change = 0.0
        total_power = 0.0
        
        # Changes
        temp_change += heater_pwr * self.heater_temp_rate
        total_power += heater_pwr * self.heater_power_consumption

        hum_change += humidifier_pwr * self.humidifier_hum_rate
        total_power += humidifier_pwr * self.humidifier_power_consumption

        temp_change += fan_pwr * self.fan_temp_rate
        hum_change += fan_pwr * self.fan_hum_rate
        total_power += fan_pwr * self.fan_power_consumption
        

        # Apply environmental disturbance (with adaptive noise)
        env_enabled = self.noise_config['env_disturbance']['enabled']
        env_std = self.noise_config['env_disturbance'].get('std', 0.05)
        adaptive_enabled = self.noise_config['adaptive_noise']['enabled']
        if adaptive_enabled:
            base_std = self.noise_config['adaptive_noise'].get('base_std', 0.05)
            scale_factor = self.noise_config['adaptive_noise'].get('scale_factor', 0.1)
            adaptive_noise_level = base_std + scale_factor * self.avg_error
            env_std += adaptive_noise_level  # Add adaptive noise to env disturbance

        if env_enabled:
            temp_change += self.np_random.normal(0, env_std)
            hum_change += self.np_random.normal(0, env_std)
        
        self.temperature += temp_change
        self.humidity += hum_change

        self.temperature = np.clip(self.temperature, 0.0, 100.0)
        self.humidity = np.clip(self.humidity, 0.0, 100.0)

        reward, temp_error, hum_error, energy_weight, energy_norm, error_norm = self.calculate_reward(total_power, action)

        # Add sensor noise to observation (with adaptive noise)
        obs_temp = self.temperature
        obs_hum = self.humidity
        sensor_enabled = self.noise_config['sensor_noise']['enabled']
        sensor_std = self.noise_config['sensor_noise'].get('std', 0.1)
        if adaptive_enabled:
            sensor_std += adaptive_noise_level  # Add adaptive noise to sensor noise

        if sensor_enabled:
            obs_temp += self.np_random.normal(0, sensor_std)
            obs_hum += self.np_random.normal(0, sensor_std)
            obs_temp = np.clip(obs_temp, 0.0, 100.0)
            obs_hum = np.clip(obs_hum, 0.0, 100.0)

        self.time += 1
        truncated = False
        terminated = self.time >= self.time_limit
        observation = np.array([obs_temp, obs_hum, self.target_temp, self.target_hum], dtype=np.float32)
        
        
        info = {
            "temp_error": temp_error,
            "hum_error": hum_error,
            "energy_weight": energy_weight,
            "energy_norm": energy_norm,
            "error_norm": error_norm,
        }
        

        return observation, reward, terminated, truncated, info

    def calculate_reward(self, total_power, action):
        temp_error = abs(self.temperature - self.target_temp)
        hum_error = abs(self.humidity - self.target_hum)

        # Update error history for adaptive noise
        current_error = (temp_error + hum_error) / 2.0
        self.error_history.append(current_error)
        if len(self.error_history) > 10:
            self.error_history.pop(0)
        if len(self.error_history) > 0:
            self.avg_error = np.mean(self.error_history)
        
        # normalization
        energy_norm = total_power / 185.0
        error_norm = (temp_error + hum_error) / 200.0
        action_change_norm = np.sum(np.abs(action - self.prev_action))

        # energy weight increases as error decreases (bounded)
        energy_weight = 2.0 + 2.0 * (1.0 - error_norm)  # ranges [2,4]
        # or piecewise if you prefer sharper effect

        reward = - energy_weight * energy_norm - 1.0 * error_norm - 0.05 * action_change_norm

        self.prev_temp_error = temp_error
        self.prev_hum_error = hum_error
        self.prev_action = action.copy()

        return reward, temp_error, hum_error, energy_weight, energy_norm, error_norm


    def _merge_noise_config(self, user_cfg):
        cfg = {}
        for k, v in self.DEFAULT_NOISE.items():
            cfg[k] = v.copy()
            if user_cfg and k in user_cfg and isinstance(user_cfg[k], dict):
                cfg[k].update(user_cfg[k])
        return cfg

    def render(self):
        print(f"Temp: {self.temperature:.2f}, Humidity: {self.humidity:.2f}")