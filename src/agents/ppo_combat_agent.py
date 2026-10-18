from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.environment.zelda_env import ZeldaEnv
from src.environment.zelda_combat_env import ZeldaCombatEnv


def make_combat_env():
    """Construit l'environnement de combat avec les obstacles existants."""

    navigation_env = ZeldaEnv(
        grid_size=8,
        max_steps=100,
    )

    obstacles = navigation_env.obstacles
    navigation_env.close()

    return ZeldaCombatEnv(
        grid_size=8,
        max_steps=100,
        player_max_hp=3,
        enemy_max_hp=3,
        obstacles=obstacles,
    )


def train_ppo(total_timesteps=300_000, seed=42):
    """Entraîne PPO à combattre un ennemi."""

    env = make_combat_env()

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

    print("\nDébut de l'entraînement PPO combat...")

    model.learn(total_timesteps=total_timesteps)

    print("Entraînement terminé.")

    Path("models").mkdir(parents=True, exist_ok=True)
    model.save("models/ppo_zelda_combat")

    env.close()

    return model


def evaluate_ppo(model, n_episodes=100, seed=1000):
    """Évalue PPO sur une série fixe de combats."""

    env = make_combat_env()

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

        if info["success"]:
            successes += 1
            successful_lengths.append(steps)
        else:
            failures += 1

    print("\n=== Résultats PPO Combat ===")
    print(f"Épisodes : {n_episodes}")
    print(f"Taux de victoire : {successes / n_episodes:.1%}")
    print(f"Combats non gagnés : {failures}")
    print(f"Récompense moyenne : {np.mean(rewards):.3f}")
    print(f"Actions moyennes : {np.mean(episode_lengths):.2f}")

    if successful_lengths:
        print(
            "Actions moyennes lors des victoires : "
            f"{np.mean(successful_lengths):.2f}"
        )

    env.close()


if __name__ == "__main__":
    model = train_ppo(
        total_timesteps=300_000,
        seed=42,
    )

    evaluate_ppo(
        model,
        n_episodes=100,
        seed=1000,
    )