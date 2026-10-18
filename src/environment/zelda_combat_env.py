from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ZeldaCombatEnv(gym.Env):
    """
    Environnement de combat 2D minimal.

    Objectif : vaincre un ennemi sans perdre tous ses PV.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size=8,
        max_steps=100,
        player_max_hp=3,
        enemy_max_hp=3,
        obstacles=None,
        render_mode=None,
    ):
        super().__init__()

        if grid_size < 2:
            raise ValueError("grid_size doit être au moins égal à 2.")

        if max_steps < 1:
            raise ValueError("max_steps doit être positif.")

        if player_max_hp < 1 or enemy_max_hp < 1:
            raise ValueError("Les PV maximum doivent être positifs.")

        if render_mode not in (None, "human"):
            raise ValueError("render_mode doit être None ou 'human'.")

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.player_max_hp = player_max_hp
        self.enemy_max_hp = enemy_max_hp
        self.render_mode = render_mode

        self.obstacles = {
            (int(x), int(y))
            for x, y in (obstacles or set())
        }

        # Vérifier que tous les obstacles sont sur la grille.
        for x, y in self.obstacles:
            if not (0 <= x < grid_size and 0 <= y < grid_size):
                raise ValueError(f"Obstacle hors de la grille : {(x, y)}")

        # Link commence en haut à gauche.
        if (0, 0) in self.obstacles:
            raise ValueError("La case de départ (0, 0) doit être libre.")

        # Actions :
        # 0 = haut, 1 = bas, 2 = gauche, 3 = droite, 4 = attaque.
        self.action_space = spaces.Discrete(5)

        # Observation :
        # [joueur_x, joueur_y, ennemi_x, ennemi_y,
        #  PV_joueur, PV_ennemi]
        # Toutes les valeurs sont normalisées entre 0 et 1.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.action_deltas = {
            0: np.array([0, -1], dtype=np.int32),
            1: np.array([0, 1], dtype=np.int32),
            2: np.array([-1, 0], dtype=np.int32),
            3: np.array([1, 0], dtype=np.int32),
        }

        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.enemy_pos = np.array([grid_size - 1, grid_size - 1],
                                  dtype=np.int32)

        self.player_hp = player_max_hp
        self.enemy_hp = enemy_max_hp
        self.steps = 0

    def _distance(self):
        """Distance de Manhattan entre Link et l'ennemi."""
        return int(np.abs(self.player_pos - self.enemy_pos).sum())

    def _get_obs(self):
        """Construit l'observation donnée à l'agent."""

        scale = self.grid_size - 1

        observation = np.array(
            [
                self.player_pos[0] / scale,
                self.player_pos[1] / scale,
                self.enemy_pos[0] / scale,
                self.enemy_pos[1] / scale,
                self.player_hp / self.player_max_hp,
                self.enemy_hp / self.enemy_max_hp,
            ],
            dtype=np.float32,
        )

        return observation

    def _reachable_cells(self):
        """
        Renvoie les cases accessibles depuis la position initiale,
        en tenant compte des obstacles.
        """

        start = (0, 0)
        reachable = {start}
        queue = deque([start])

        while queue:
            x, y = queue.popleft()

            for delta in self.action_deltas.values():
                nx = x + int(delta[0])
                ny = y + int(delta[1])
                cell = (nx, ny)

                inside_grid = (
                    0 <= nx < self.grid_size
                    and 0 <= ny < self.grid_size
                )

                if (
                    inside_grid
                    and cell not in self.obstacles
                    and cell not in reachable
                ):
                    reachable.add(cell)
                    queue.append(cell)

        return sorted(reachable)

    def reset(self, seed=None, options=None):
        """Démarre un nouvel épisode de combat."""

        super().reset(seed=seed)

        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.player_hp = self.player_max_hp
        self.enemy_hp = self.enemy_max_hp
        self.steps = 0

        # L'ennemi apparaît sur une case libre et accessible.
        candidates = [
            cell
            for cell in self._reachable_cells()
            if cell != (0, 0)
        ]

        if not candidates:
            raise ValueError(
                "Aucune case accessible pour placer l'ennemi."
            )

        index = int(self.np_random.integers(len(candidates)))
        self.enemy_pos = np.array(candidates[index], dtype=np.int32)

        observation = self._get_obs()

        info = {
            "player_hp": self.player_hp,
            "enemy_hp": self.enemy_hp,
            "distance": self._distance(),
            "success": False,
        }

        return observation, info

    def step(self, action):
        """Exécute une action et calcule la récompense."""

        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide : {action}")

        if self.player_hp <= 0 or self.enemy_hp <= 0:
            raise RuntimeError(
                "L'épisode est terminé. Appelle reset() avant step()."
            )

        old_distance = self._distance()
        reward = -0.01

        # -------------------------
        # 1. Action de Link
        # -------------------------

        if action in self.action_deltas:
            delta = self.action_deltas[int(action)]

            candidate = np.clip(
                self.player_pos + delta,
                0,
                self.grid_size - 1,
            ).astype(np.int32)

            candidate_cell = (
                int(candidate[0]),
                int(candidate[1]),
            )

            # On ne peut traverser ni un mur ni l'ennemi.
            if (
                candidate_cell not in self.obstacles
                and not np.array_equal(candidate, self.enemy_pos)
            ):
                self.player_pos = candidate

        else:
            # Action 4 : attaquer.
            if self._distance() == 1:
                self.enemy_hp -= 1
                reward += 0.25

                if self.enemy_hp == 0:
                    reward += 2.0
            else:
                # Une attaque hors de portée ne touche pas.
                reward -= 0.05

        # Récompenser le rapprochement de l'ennemi.
        new_distance = self._distance()
        reward += 0.05 * (old_distance - new_distance)

        # -------------------------
        # 2. Riposte de l'ennemi
        # -------------------------

        # L'ennemi riposte uniquement si Link l'a attaqué
        # au corps-à-corps et qu'il est encore vivant.

        if (
            action == 4
            and self.enemy_hp > 0
            and self._distance() == 1
        ):
            self.player_hp -= 1
            reward -= 0.25

            if self.player_hp == 0:
                reward -= 2.0
        # -------------------------
        # 3. Fin de l'épisode
        # -------------------------

        self.steps += 1

        terminated = (
            self.enemy_hp <= 0
            or self.player_hp <= 0
        )

        truncated = (
            self.steps >= self.max_steps
            and not terminated
        )

        success = self.enemy_hp <= 0

        observation = self._get_obs()

        info = {
            "player_hp": self.player_hp,
            "enemy_hp": self.enemy_hp,
            "distance": self._distance(),
            "steps": self.steps,
            "success": success,
            "player_pos": tuple(int(v) for v in self.player_pos),
            "enemy_pos": tuple(int(v) for v in self.enemy_pos),
        }

        return observation, float(reward), terminated, truncated, info

    def render(self):
        """Affiche la grille et les PV dans le terminal."""

        if self.render_mode != "human":
            return

        for y in range(self.grid_size):
            row = []

            for x in range(self.grid_size):
                cell = (x, y)

                if np.array_equal(self.player_pos, [x, y]):
                    row.append("P")
                elif np.array_equal(self.enemy_pos, [x, y]):
                    row.append("E")
                elif cell in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")

            print(" ".join(row))

        print(
            f"Link : {self.player_hp}/{self.player_max_hp} PV | "
            f"Ennemi : {self.enemy_hp}/{self.enemy_max_hp} PV"
        )
        print("P = joueur | E = ennemi | # = obstacle\n")