from src.environment.zelda_env import ZeldaEnv


def play_episode(env, seed=None, render=False):
    """Joue un épisode en choisissant les actions au hasard."""

    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    if render:
        print("État initial :")
        env.render()

    while True:
        # L'agent choisit une action aléatoire.
        action = env.action_space.sample()

        # Le jeu exécute l'action et retourne le résultat.
        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        steps += 1

        if render:
            print(
                f"Action : {action} | "
                f"Récompense : {reward:.2f} | "
                f"Distance : {info['distance']}"
            )
            env.render()

        # L'épisode s'arrête si la cible est atteinte
        # ou si la limite de pas est dépassée.
        if terminated or truncated:
            break

    return total_reward, steps, terminated


def evaluate_random_agent(n_episodes=100):
    """Évalue l'agent aléatoire sur plusieurs épisodes."""

    env = ZeldaEnv(grid_size=5, max_steps=50)

    # Pour rendre les expériences reproductibles.
    env.action_space.seed(42)

    rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(n_episodes):
        reward, steps, terminated = play_episode(
            env,
            seed=42 + episode,
            render=False,
        )

        rewards.append(reward)
        episode_lengths.append(steps)

        if terminated:
            successes += 1

    env.close()

    print("\n=== Résultats de l'agent aléatoire ===")
    print(f"Nombre d'épisodes : {n_episodes}")
    print(f"Taux de réussite : {successes / n_episodes:.1%}")
    print(f"Récompense moyenne : {sum(rewards) / n_episodes:.3f}")
    print(
        f"Nombre moyen de pas : "
        f"{sum(episode_lengths) / n_episodes:.2f}"
    )


if __name__ == "__main__":
    evaluate_random_agent()