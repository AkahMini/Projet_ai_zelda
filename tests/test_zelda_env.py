from gymnasium.utils.env_checker import check_env
from src.environment.zelda_env import ZeldaEnv


def main():
    # 1. Vérifier la conformité de l'environnement Gymnasium.
    env_check = ZeldaEnv()
    check_env(env_check)
    env_check.close()

    print("Environnement Gymnasium valide !")

    # 2. Tester des actions aléatoires et afficher la grille.
    env = ZeldaEnv(
        grid_size=5,
        max_steps=20,
        render_mode="human",
    )

    observation, info = env.reset(seed=42)

    print("Observation initiale :", observation)
    print("Informations :", info)
    env.render()

    for step in range(10):
        # L'agent choisit une action aléatoire.
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        print(
            f"Étape {step + 1} | "
            f"Action : {action} | "
            f"Récompense : {reward:.2f} | "
            f"Distance : {info['distance']}"
        )

        env.render()

        if terminated or truncated:
            print(
                "Objectif atteint !"
                if terminated
                else "Limite de pas atteinte."
            )
            break

    env.close()


if __name__ == "__main__":
    main()