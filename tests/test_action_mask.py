import numpy as np

from src.environment.zelda_combat_env import ZeldaCombatEnv
from src.environment.zelda_env import ZeldaEnv


def main():

    navigation_env = ZeldaEnv(grid_size=8)

    env = ZeldaCombatEnv(
        grid_size=8,
        max_steps=100,
        obstacles=navigation_env.obstacles,
    )

    navigation_env.close()

    env.reset(seed=42)

    # Cas 1 : ennemi en diagonale.
    env.player_pos = np.array([0, 0], dtype=np.int32)
    env.enemy_pos = np.array([1, 1], dtype=np.int32)

    masks = env.action_masks()

    print("=== Ennemi en diagonale ===")
    print("Masque :", masks)

    assert masks.shape == (5,)
    assert masks.dtype == np.bool_
    assert not masks[4], "L'attaque devrait être interdite."

    # Cas 2 : ennemi directement à droite.
    env.player_pos = np.array([0, 0], dtype=np.int32)
    env.enemy_pos = np.array([1, 0], dtype=np.int32)

    masks = env.action_masks()

    print("\n=== Ennemi à droite ===")
    print("Masque :", masks)

    assert masks[4], "L'attaque devrait être autorisée."
    assert not masks[3], "La case occupée par l'ennemi doit être interdite."

    print("\nTous les tests du masque sont réussis !")

    env.close()


if __name__ == "__main__":
    main()