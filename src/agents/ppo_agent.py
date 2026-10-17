from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.environment.zelda_env import ZeldaEnv


def train_ppo(total_timesteps=50_000, seed=42):
    """Entraîne un agent PPO sur notre environnement."""

    env = ZeldaEnv(
        grid_size=5,
        max_steps=50,
    )

    # Création de l'agent PPO.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    print("\nDébut de l'entraînement PPO...")

    # total_timesteps correspond au nombre de transitions
    # collectées dans l'environnement, pas au nombre d'épisodes.
    model.learn(total_timesteps=total_timesteps)

    print("\nEntraînement terminé !")

    # Sauvegarde du modèle entraîné.
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(models_dir / "ppo_zelda_grid"))

    env.close()

    return model


def evaluate_ppo(model, n_episodes=100, seed=42):
    """Évalue l'agent PPO après son entraînement."""

    env = ZeldaEnv(
        grid_size=5,
        max_steps=50,
    )

    rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(n_episodes):

        # Chaque épisode reçoit une graine déterminée.
        observation, info = env.reset(seed=seed + episode)

        total_reward = 0.0
        steps = 0

        while True:

            # En évaluation, on choisit l'action la plus probable.
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(int(action))
            )

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        episode_lengths.append(steps)

        if terminated:
            successes += 1

    env.close()

    print("\n=== Résultats de l'agent PPO ===")
    print(f"Nombre d'épisodes : {n_episodes}")
    print(f"Taux de réussite : {successes / n_episodes:.1%}")
    print(f"Récompense moyenne : {np.mean(rewards):.3f}")
    print(f"Nombre moyen de pas : {np.mean(episode_lengths):.2f}")

    return {
        "success_rate": successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(episode_lengths)),
    }


if __name__ == "__main__":

    # 1. Entraîner l'agent.
    model = train_ppo(
        total_timesteps=50_000,
        seed=42,
    )

    # 2. Évaluer le modèle entraîné.
    evaluate_ppo(
        model,
        n_episodes=100,
        seed=42,
    )