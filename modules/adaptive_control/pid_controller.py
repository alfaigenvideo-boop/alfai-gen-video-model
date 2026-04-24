import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PIDConfig:
    """PID kontrolcüsü için hiperparametre konfigürasyonu."""
    kp: float = 0.8
    ki: float = 0.05
    kd: float = 0.1
    integral_limit: float = 0.5
    output_limit: Tuple[float, float] = (0.3, 1.0)

class PIDController:
    """
    Oransal-İntegral-Türevsel (PID) Kontrolcüsü.
    Hedeflenen değer ile gerçekleşen değer arasındaki hatayı minimize eder.
    """
    def __init__(self, config: PIDConfig, initial_value: float):
        self.config = config
        self.value = initial_value
        self.integral = 0.0
        self.previous_error = 0.0

    def update(self, error: float) -> float:
        """Hataya göre PID delta (değişim) miktarını hesaplar."""
        # Oransal (Proportional)
        proportional = self.config.kp * error
        
        # İntegral (Anti-windup ile birlikte)
        self.integral += error
        self.integral = np.clip(
            self.integral,
            -self.config.integral_limit,
            self.config.integral_limit
        )
        integral_term = self.config.ki * self.integral
        
        # Türevsel (Derivative)
        derivative = error - self.previous_error
        derivative_term = self.config.kd * derivative
        
        output = proportional + integral_term + derivative_term
        self.previous_error = error
        
        return output

    def adjust_value(self, delta: float) -> float:
        """Hesaplanan delta'yı mevcut değere uygulayıp limitler içinde tutar."""
        self.value += delta
        self.value = np.clip(
            self.value,
            self.config.output_limit[0],
            self.config.output_limit[1]
        )
        return self.value
