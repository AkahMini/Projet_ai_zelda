from collections import Counter, deque

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
        """Calcule la longueur du chemin le plus court vers l'ennemi,
        en tenant compte des obstacles.
        """
        start = tuple(int(v) for v in self.player_pos)
        goal = tuple(int(v) for v in self.enemy_pos)

        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (x, y), distance = queue.popleft()

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
                    and cell not in visited
                ):
                    if cell == goal:
                        return distance + 1

                    visited.add(cell)
                    queue.append((cell, distance + 1))

        # Normalement impossible puisque l'ennemi est placé
        # sur une case accessible depuis le départ.
        return self.grid_size * self.grid_size

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
        """Exécute l'action de Link puis celle de l'ennemi."""

        if not self.action_space.contains(action):
            raise ValueError(f"Action invalide : {action}")

        if self.player_hp <= 0 or self.enemy_hp <= 0:
            raise RuntimeError(
                "L'épisode est terminé. Appelle reset() avant step()."
            )

        old_distance = self._distance()
        reward = -0.01
        # Vérifier si l'ennemi était déjà adjacent
        # avant l'action de Link.
        enemy_was_adjacent = self._is_adjacent()

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

            if (
                candidate_cell not in self.obstacles
                and not np.array_equal(candidate, self.enemy_pos)
                and not np.array_equal(candidate, self.player_pos)
            ):
                self.player_pos = candidate

            else:
                reward -= 0.05

        else:
            # Action 4 : attaque
            if self._is_adjacent():

                self.enemy_hp -= 1
                reward += 0.5

                if self.enemy_hp <= 0:
                    reward += 2.0

            else:
                reward -= 0.05

        # Récompense uniquement le déplacement de Link
        distance_after_link = self._distance()

        reward += 0.05 * (
            old_distance - distance_after_link
        )

        # -------------------------
        # 2. Action de l'ennemi
        # -------------------------

        enemy_moved = False
        enemy_attacked = False

        if self.enemy_hp > 0 and self.player_hp > 0:

            # L'ennemi peut attaquer uniquement s'il
            # était déjà adjacent avant l'action de Link
            # et s'il est toujours adjacent après celle-ci.

            if enemy_was_adjacent and self._is_adjacent():

                self.player_hp -= 1
                reward -= 0.25
                enemy_attacked = True

                if self.player_hp <= 0:
                    reward -= 2.0

            # Si l'ennemi n'est plus adjacent, il poursuit Link.
            elif not self._is_adjacent():

                enemy_moved = self._move_enemy_towards_player()

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
            "player_pos": tuple(
                int(v) for v in self.player_pos
            ),
            "enemy_pos": tuple(
                int(v) for v in self.enemy_pos
            ),
            "enemy_moved": enemy_moved,
            "enemy_attacked": enemy_attacked,
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

    def _is_adjacent(self):
        """Vérifie si l'ennemi est à une case de Link."""

        distance = np.abs(
            self.player_pos - self.enemy_pos
        ).sum()

        return distance == 1

    def action_masks(self):
        """
        Retourne un masque booléen des actions autorisées.

        True  = action autorisée
        False = action interdite
        """

        # Cinq actions possibles.
        masks = np.zeros(5, dtype=bool)

        # Vérifier les quatre déplacements.
        for action, delta in self.action_deltas.items():

            candidate = self.player_pos + delta

            x = int(candidate[0])
            y = int(candidate[1])

            candidate_cell = (x, y)

            # Vérifier que la position est dans la grille.
            inside_grid = (
                0 <= x < self.grid_size
                and 0 <= y < self.grid_size
            )

            # Vérifier que la case n'est pas un obstacle.
            not_obstacle = candidate_cell not in self.obstacles

            # Vérifier que la case n'est pas occupée par l'ennemi.
            not_enemy = not np.array_equal(
                candidate,
                self.enemy_pos,
            )

            # Autoriser uniquement les déplacements possibles.
            masks[action] = (
                inside_grid
                and not_obstacle
                and not_enemy
            )

        # Autoriser l'attaque uniquement si l'ennemi est adjacent.
        masks[4] = (
            self.enemy_hp > 0
            and self._is_adjacent()
        )

        # Un état sans aucune action autorisée signale un problème
        # de configuration de la carte ou de l'environnement.
        if not masks.any():
            raise RuntimeError(
                f"Aucune action possible : "
                f"Link={self.player_pos}, "
                f"Ennemi={self.enemy_pos}"
            )

        return masks

    def _move_enemy_towards_player(self):
        """Déplace l'ennemi d'une case vers Link en évitant les obstacles."""

        # Position actuelle de l'ennemi et position de Link
        start = tuple(int(v) for v in self.enemy_pos)
        goal = tuple(int(v) for v in self.player_pos)

        # Initialisation du BFS
        queue = deque([start])
        previous = {start: None}

        # Recherche du chemin le plus court
        while queue and goal not in previous:

            current = queue.popleft()

            # Examiner les quatre déplacements possibles
            for delta in self.action_deltas.values():

                next_cell = (
                    current[0] + int(delta[0]),
                    current[1] + int(delta[1])
                )

                x, y = next_cell

                # Vérifier les limites de la grille
                if not (0 <= x < self.grid_size
                        and 0 <= y < self.grid_size):
                    continue

                # Éviter les obstacles
                if next_cell in self.obstacles:
                    continue

                # Ne pas revisiter une case
                if next_cell in previous:
                    continue

                previous[next_cell] = current
                queue.append(next_cell)

                if next_cell == goal:
                    break

        # Aucun chemin trouvé
        if goal not in previous:
            return False

        # Reconstruire le chemin entre l'ennemi et Link
        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = previous[current]

        path.reverse()

        # Si Link est déjà adjacent, l'ennemi ne bouge pas
        if len(path) <= 2:
            return False

        # Faire avancer l'ennemi d'une seule case
        next_position = path[1]

        self.enemy_pos = np.array(
            next_position,
            dtype=np.int32
        )

        return True