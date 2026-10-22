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


def train_ppo(total_timesteps=300_000, seed=789):
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
    timeout_details = []

    deaths = 0
    timeouts = 0
    successes = 0

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

        elif info["player_hp"] <= 0:
            deaths += 1

        elif truncated:
            timeouts += 1

            timeout_details.append({
                "episode": episode,
                "player_pos": info["player_pos"],
                "enemy_pos": info["enemy_pos"],
                "distance": info["distance"],
                "player_hp": info["player_hp"],
                "enemy_hp": info["enemy_hp"],
                "steps": steps,
            })

    print("\n=== Résultats PPO Combat ===")
    print(f"Épisodes : {n_episodes}")
    print(f"Taux de victoire : {successes / n_episodes:.1%}")
    print(f"Morts de Link : {deaths}")
    print(f"Limites de temps atteintes : {timeouts}")
    print(f"Récompense moyenne : {np.mean(rewards):.3f}")
    print(f"Actions moyennes : {np.mean(episode_lengths):.2f}")

    if successful_lengths:
        print(
            "Actions moyennes lors des victoires : "
            f"{np.mean(successful_lengths):.2f}"
        )

    print("\n=== Analyse des timeouts ===")

    for result in timeout_details[:10]:
        print(
            f"Épisode {result['episode']} | "
            f"Link {result['player_pos']} | "
            f"Ennemi {result['enemy_pos']} | "
            f"Distance {result['distance']} | "
            f"PV Link {result['player_hp']} | "
            f"PV ennemi {result['enemy_hp']} | "
            f"Actions {result['steps']}"
        )

    env.close()


if __name__ == "__main__":
    model = train_ppo(
        total_timesteps=300_000,
        seed=1000,
    )

    evaluate_ppo(
        model,
        n_episodes=100,
        seed=1000,
    )