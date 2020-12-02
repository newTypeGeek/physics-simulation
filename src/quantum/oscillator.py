import numpy as np
from scipy.special import eval_hermite


class QuantumHarmonicOscillator:
    """
    1D Quantum Harmonic Oscillator

    For simplicity, we use a unit with hbar = 1

    """
    __hbar = 1

    def __init__(self, mass: float, angular_freq: float):
        assert mass > 0, "Mass m must be positive real number"
        assert angular_freq > 0, "Angular frequency omega must be positive real number"
        self.__mass = mass
        self.__angular_freq = angular_freq

    @property
    def mass_(self):
        return self.__mass

    @property
    def angular_freq_(self):
        return self.__angular_freq

    def compute_eigen_energy(self, n: int) -> float:
        assert n >= 0, "Quantum number n must be non negative integer"
        return self.__hbar * self.__angular_freq * (n + 0.5)

    def compute_energy_eigen_state_amplitude(self, n: int, x_grid: np.array) -> np.array:
        assert n >= 0, "Quantum number n must be non negative integer"

        _common_factor = (self.__mass * self.__angular_freq) / self.__hbar
        _factor_1 = np.power(_common_factor / np.pi, 0.25) / np.sqrt((2**n) * np.math.factorial(n))
        _factor_2 = np.exp(-_common_factor*(x_grid ** 2) / 2)
        hermite_args = np.sqrt(_common_factor) * x_grid
        return _factor_1 * _factor_2 * self.__hermite_poly(hermite_args, n)

    def compute_amplitude_evolution_delta_init(self,
                                               x_grid: np.array,
                                               t_grid: np.array,
                                               x_init: float,
                                               n_max: int = 50) -> np.array:

        assert n_max >= 0, f"Number of (eigen) term n_max must be non-negative integer"

        _common_factor = (self.__mass * self.__angular_freq) / self.__hbar
        hermite_args_init = np.sqrt(_common_factor) * x_init
        hermite_args = np.sqrt(_common_factor) * x_grid
        _factor_2 = np.exp(-_common_factor*(x_grid**2 + x_init**2) / 2)

        spatial_prods = []
        eigen_energies = []
        for n in range(n_max+1):
            _factor_1 = np.sqrt(_common_factor / np.pi) / ((2**n) * np.math.factorial(n))
            hermite_prod = self.__hermite_poly(hermite_args, n) * self.__hermite_poly(hermite_args_init, n)

            # 1. Product factor with spatial part only
            spatial_prod = _factor_1 * _factor_2 * hermite_prod

            # 2. Compute the plane wave part
            eigen_energy = self.compute_eigen_energy(n)

            spatial_prods.append(spatial_prod)
            eigen_energies.append(eigen_energy)

        # Split to another loop to avoid repeated computation
        wavefunction = []
        for t in t_grid:
            _sum = 0.
            for spatial_prod, eigen_energy in zip(spatial_prods, eigen_energies):
                plane_wave = np.exp(complex(0, -eigen_energy * t / self.__hbar))
                _sum += spatial_prod * plane_wave

            wavefunction.append(_sum)

        return np.array(wavefunction)

    @staticmethod
    def __hermite_poly(x: np.array, n: int) -> np.array:
        return eval_hermite(n, x)


