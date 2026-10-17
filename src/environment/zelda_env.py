import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ZeldaEnv(gym.Env):
    #Env 2d simple pour débuter

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=5, max_steps=50, render_mode=None):
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size doit être au moins égal à 2.")

        if max_steps < 1:
            raise ValueError("max_steps doit être positif.")

        if render_mode not in (None, "human"):
            raise ValueError("render_mode doit être None ou 'human'.")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Quatre actions possibles : haut, bas, gauche, droite.
        self.action_space = spaces.Discrete(4)

        # Observation : x et y du joueur, puis x et y de la cible.
        # Les coordonnées sont normalisées entre 0 et 1.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.target_pos = np.array([1, 1], dtype=np.int32)
        self.steps = 0

        # Déplacements associés aux actions.
        self.action_deltas = {
            0: np.array([0, -1], dtype=np.int32),  # Haut
            1: np.array([0,  1], dtype=np.int32),  # Bas
            2: np.array([-1, 0], dtype=np.int32),  # Gauche
            3: np.array([ 1, 0], dtype=np.int32),  # Droite
        }

    def _get_obs(self):
        #Retourne les coordonnées normalisées du joueur et de la cible
        coordinates = np.concatenate(
            (self.player_pos, self.target_pos)
        ).astype(np.float32)
        return coordinates / (self.grid_size - 1)

    def _get_distance(self):
        return int(np.abs(self.player_pos - self.target_pos).sum())

    def reset(self, seed=None, options=None):
        """Démarre un nouvel épisode."""
        super().reset(seed=seed)

        # Le joueur commence toujours en haut à gauche.
        self.player_pos = np.array([0, 0], dtype=np.int32)

        # La cible est placée aléatoirement, sauf sur le joueur.
        while True:
            self.target_pos = self.np_random.integers(
                low=0,
                high=self.grid_size,
                size=2,
                dtype=np.int32,
            )

            if not np.array_equal(self.player_pos, self.target_pos):
                break

        self.steps = 0

        observation = self._get_obs()
        info = {"distance": self._get_distance()}

        return observation, info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide : {action}")

        old_distance = self._get_distance()

        # Déplacement du joueur.
        delta = self.action_deltas[int(action)]
        self.player_pos = np.clip(
            self.player_pos + delta,
            0,
            self.grid_size - 1,
        ).astype(np.int32)

        self.steps += 1

        new_distance = self._get_distance()

        # Une action coûte légèrement quelque chose.
        reward = -0.01

        # Récompenser le rapprochement et pénaliser l'éloignement.
        reward += 0.1 * (old_distance - new_distance)

        # Bonus important lorsque la cible est atteinte.
        terminated = np.array_equal(
            self.player_pos,
            self.target_pos,
        )

        if terminated:
            reward += 1.0

        # Limite de durée d'un épisode.
        truncated = self.steps >= self.max_steps and not terminated

        observation = self._get_obs()
        info = {
            "distance": new_distance,
            "steps": self.steps,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Affiche la grille dans le terminal."""
        if self.render_mode != "human":
            return

        rows = []

        for y in range(self.grid_size):
            row = []

            for x in range(self.grid_size):
                if np.array_equal(self.player_pos, [x, y]):
                    row.append("P")
                elif np.array_equal(self.target_pos, [x, y]):
                    row.append("T")
                else:
                    row.append(".")

            rows.append(" ".join(row))

        print("\n".join(rows))
        print("P = joueur | T = cible\n")