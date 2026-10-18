from gymnasium.utils.env_checker import check_env

from src.environment.zelda_env import ZeldaEnv
from src.environment.zelda_combat_env import ZeldaCombatEnv


def main():
    # Réutiliser exactement les obstacles de l'environnement actuel.
    navigation_env = ZeldaEnv(
        grid_size=8,
        max_steps=100,
    )

    obstacles = navigation_env.obstacles

    env_check = ZeldaCombatEnv(
        grid_size=8,
        max_steps=100,
        obstacles=obstacles,
    )

    # Vérifier que l'environnement respecte Gymnasium.
    check_env(env_check)
    env_check.close()

    print("Environnement de combat Gymnasium valide !")

    # Faire une démonstration aléatoire.
    env = ZeldaCombatEnv(
        grid_size=8,
        max_steps=100,
        obstacles=obstacles,
        render_mode="human",
    )

    observation, info = env.reset(seed=42)

    print("Observation initiale :", observation)
    print("Informations initiales :", info)
    env.render()

    while True:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        print(
            f"Action : {action} | "
            f"Récompense : {reward:.2f} | "
            f"PV Link : {info['player_hp']} | "
            f"PV ennemi : {info['enemy_hp']}"
        )

        env.render()

        if terminated or truncated:
            if info["success"]:
                print("Victoire ! L'ennemi est vaincu.")
            elif terminated:
                print("Défaite ! Link n'a plus de PV.")
            else:
                print("Temps écoulé !")

            break

    env.close()
    navigation_env.close()


if __name__ == "__main__":
    main()