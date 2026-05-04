from abc import ABC, abstractmethod

import numpy as np

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False


class InputHandler(ABC):

    @abstractmethod
    def get_action(self) -> np.ndarray:
        """Return action array of shape (3,) with values in {0, 1, 2}."""

    def close(self):
        pass


def _default_player1_keys():
    if not _PYGAME_AVAILABLE:
        return {}
    return {
        pygame.K_w:     (0, 1),   # forward
        pygame.K_s:     (0, 2),   # backward
        pygame.K_a:     (1, 1),   # strafe left
        pygame.K_d:     (1, 2),   # strafe right
        pygame.K_q:     (2, 1),   # rotate left
        pygame.K_e:     (2, 2),   # rotate right
    }


def _default_player2_keys():
    if not _PYGAME_AVAILABLE:
        return {}
    return {
        pygame.K_UP:     (0, 1),
        pygame.K_DOWN:   (0, 2),
        pygame.K_LEFT:   (1, 1),
        pygame.K_RIGHT:  (1, 2),
        pygame.K_COMMA:  (2, 1),   # < key
        pygame.K_PERIOD: (2, 2),   # > key
    }


PLAYER1_KEYS = _default_player1_keys()
PLAYER2_KEYS = _default_player2_keys()


class KeyboardInputHandler(InputHandler):

    def __init__(self, key_mapping: dict):
        if not _PYGAME_AVAILABLE:
            raise ImportError("pygame is required for KeyboardInputHandler. pip install pygame")
        self.key_mapping = key_mapping

    def get_action(self) -> np.ndarray:
        keys   = pygame.key.get_pressed()
        action = [0, 0, 0]
        for key, (dim, val) in self.key_mapping.items():
            if keys[key]:
                action[dim] = val
        return np.array(action, dtype=np.int32)


class RandomInputHandler(InputHandler):
    """Fallback random policy — useful for testing collect_demos.py without a human."""

    def __init__(self, action_dims=(3, 3, 3)):
        self._dims = action_dims

    def get_action(self) -> np.ndarray:
        return np.array([np.random.randint(d) for d in self._dims], dtype=np.int32)