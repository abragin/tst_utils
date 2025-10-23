import numpy as np
from typing import List

class BaseLengthSampler:
    def sample(self, n: int) -> np.ndarray:
        raise NotImplementedError
    def expected_mean(self) -> float:
        raise NotImplementedError

class GammaLengthSampler(BaseLengthSampler):
    """Gamma-distributed chunk lengths truncated to [min_len, max_len]."""
    def __init__(
        self, mean_len=64, min_len=32, max_len=128,
        shape=1.5
    ):
        self.shape = shape
        self.scale = 1 / shape  # ensures mean=1
        self.mean_len = mean_len
        self.min_len = min_len
        self.max_len = max_len

    def sample(self, n):
        raw = np.random.gamma(
            shape=self.shape, scale=self.scale, size=n
        )
        # rescale so mean ~ mean_tok_len before clipping
        scaled = self.min_len + (self.mean_len - self.min_len) * raw
        clipped = np.clip(scaled, self.min_len, self.max_len)
        return clipped

    def expected_mean(self):
        return self.mean_len


class UniformLengthSampler(BaseLengthSampler):
    def __init__(self, min_len = 32, max_len = 128):
        self.min_len = min_len
        self.max_len = max_len

    def sample(self, n: int) -> np.ndarray:
        return np.random.uniform(self.min_len, self.max_len, n)

    def expected_mean(self) -> float:
        return 0.5 * (self.min_len + self.max_len)


class LogUniformLengthSampler(BaseLengthSampler):
    """Continuous distribution ~ 1/x on [low, high]."""
    def __init__(self, min_len = 32, max_len = 128):
        assert min_len > 0
        self.min_len = min_len
        self.max_len = max_len
        self._log_min_len = np.log(min_len)
        self._log_max_len = np.log(max_len)

    def sample(self, n: int) -> np.ndarray:
        u = np.random.uniform(self._log_min_len, self._log_max_len, n)
        return np.exp(u)

    def expected_mean(self) -> float:
        return (
            (self.max_len - self.min_len) /
            np.log(self.max_len / self.min_len)
        )


class MixtureLengthSampler(BaseLengthSampler):
    """Weighted mixture of sub-samplers."""
    def __init__(self, samplers: List[BaseLengthSampler], weights: List[float]):
        assert len(samplers) == len(weights)
        self.samplers = samplers
        self.weights = np.array(weights, dtype=float) / np.sum(weights)

    def sample(self, n: int) -> np.ndarray:
        n_parts = np.random.multinomial(n, self.weights)
        samples = [s.sample(k) for s, k in zip(self.samplers, n_parts) if k > 0]
        vals = np.concatenate(samples)
        np.random.shuffle(vals)   # <- shuffle to remove ordering bias
        return vals

    def expected_mean(self) -> float:
        return float(np.sum([w * s.expected_mean() for w, s in zip(self.weights, self.samplers)]))