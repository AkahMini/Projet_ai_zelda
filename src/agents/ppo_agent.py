from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.environment.zelda_env import ZeldaEnv


def train_ppo(env, total_timesteps=50_000, seed=123):
    """Entraîne un agent PPO sur notre environnement."""

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


def evaluate_ppo(model, env, n_episodes=100, seed=123):
    """Évalue l'agent PPO sur plusieurs épisodes."""

    rewards = []
    episode_lengths = []
    successful_lengths = []

    successes = 0
    failures = 0

    for episode in range(n_episodes):
        observation, info = env.reset(seed=seed + episode)

        total_reward = 0.0
        steps = 0

        while True:
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
            successful_lengths.append(steps)
        else:
            failures += 1

    success_rate = successes / n_episodes

    print("\n=== Résultats de l'agent PPO ===")
    print(f"Nombre d'épisodes : {n_episodes}")
    print(f"Taux de réussite : {success_rate:.1%}")
    print(f"Épisodes sans réussite : {failures}")
    print(f"Récompense moyenne : {np.mean(rewards):.3f}")

    print(
        f"Pas moyens (tous les épisodes) : "
        f"{np.mean(episode_lengths):.2f}"
    )

    if successful_lengths:
        print(
            f"Pas moyens (réussites uniquement) : "
            f"{np.mean(successful_lengths):.2f}"
        )
    else:
        print("Aucune réussite : moyenne des pas réussis indisponible.")

    return {
        "success_rate": success_rate,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(episode_lengths)),
        "mean_success_steps": (
            float(np.mean(successful_lengths))
            if successful_lengths else None
        ),
        "failures": failures,
    }


if __name__ == "__main__":

    eval_env = ZeldaEnv(
        grid_size=8,
        max_steps=50,
    )

    # 1. Entraîner l'agent.
    model = train_ppo(
        env=eval_env,
        total_timesteps=250_000,
        seed=42
    )

    evaluate_ppo(
        model,
        eval_env,
        n_episodes=100,
        seed=42,
    )

    eval_env.close()