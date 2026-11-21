from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from src.environment.zelda_env import ZeldaEnv
from src.environment.zelda_combat_env import ZeldaCombatEnv
from collections import Counter


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


def train_ppo(total_timesteps=300_000, seed=789, ent_coef=0.03):
    """Entraîne PPO à combattre un ennemi."""

    env = make_combat_env()

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        ent_coef=ent_coef,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    print("\nDébut de l'entraînement PPO combat...")

    model.learn(total_timesteps=total_timesteps)

    print("Entraînement terminé.")

    Path("models").mkdir(parents=True, exist_ok=True)

    model_path = f"models/maskable_ppo_aggressive_combat_v3_seed_{seed}"

    model.save(model_path)

    print(f"Modèle sauvegardé : {model_path}")

    env.close()

    return model


def evaluate_ppo(model, n_episodes=100, seed=1000, deterministic=True):
    """Évalue MaskablePPO sur une série fixe de combats."""

    print(f"\nMode déterministe : {deterministic}")

    env = make_combat_env()

    rewards = []
    episode_lengths = []
    successful_lengths = []

    timeout_details = []
    death_details = []

    deaths = 0
    timeouts = 0
    successes = 0
    total_enemy_moves = 0
    total_enemy_attacks = 0

    for episode in range(n_episodes):

        observation, info = env.reset(seed=seed + episode)

        episode_enemy_moves = 0
        total_reward = 0.0
        steps = 0

        actions_taken = []
        trace = []

        while True:

            # État avant l'action
            player_before = tuple(int(v) for v in env.player_pos)
            enemy_before = tuple(int(v) for v in env.enemy_pos)
            distance_before = int(env._distance())

            # Calcul du masque
            action_mask = env.action_masks()

            # Calcul des probabilités de chaque action
            obs_tensor, _ = model.policy.obs_to_tensor(observation)

            distribution = model.policy.get_distribution(
                obs_tensor,
                action_masks=action_mask
            )

            action_probs = (
                distribution.distribution.probs
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )

            # Prédiction de MaskablePPO
            action, _ = model.predict(
                observation,
                deterministic=deterministic,
                action_masks=action_mask
            )

            action = int(np.asarray(action).item())

            # Exécution de l'action
            observation, reward, terminated, truncated, info = env.step(action)

            # Compter les déplacements et attaques de l'ennemi
            if info.get("enemy_moved", False):
                episode_enemy_moves += 1

            if info.get("enemy_attacked", False):
                total_enemy_attacks += 1

            # Mise à jour des statistiques
            total_reward += float(reward)
            steps += 1
            actions_taken.append(action)

            # État après l'action
            player_after = tuple(int(v) for v in env.player_pos)
            distance_after = int(env._distance())

            # Enregistrement de la trace
            trace.append({
                "player_before": player_before,
                "enemy": enemy_before,
                "distance_before": distance_before,
                "mask": action_mask.tolist(),
                "action_probs": action_probs.tolist(),
                "action": action,
                "reward": float(reward),
                "player_after": player_after,
                "distance_after": distance_after,
                "enemy_moved": info.get("enemy_moved", False),
                "enemy_attacked": info.get("enemy_attacked", False)
            })

            # Vérifier la fin de l'épisode
            if terminated or truncated:
                break

        # ---------------------------------
        # Analyse des résultats de l'épisode
        # ---------------------------------

        rewards.append(total_reward)
        episode_lengths.append(steps)

        total_enemy_moves += episode_enemy_moves

        if info.get("success", False):

            successes += 1
            successful_lengths.append(steps)

        elif info.get("player_hp", env.player_hp) <= 0:

            deaths += 1

            death_details.append({
                "episode": episode,
                "player_pos": info["player_pos"],
                "enemy_pos": info["enemy_pos"],
                "player_hp": info["player_hp"],
                "enemy_hp": info["enemy_hp"],
                "steps": steps,
                "action_counts": dict(Counter(actions_taken)),
                "last_actions": actions_taken[-20:],
                "trace": trace[-20:]
            })

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
                "action_counts": dict(Counter(actions_taken)),
                "last_actions": actions_taken[-20:],
                "trace": trace[-10:]
            })

    # ---------------------------------
    # Résultats globaux
    # ---------------------------------

    print("\n=== Résultats PPO Combat ===")
    print(f"Épisodes : {n_episodes}")
    print(f"Taux de victoire : {successes / n_episodes:.1%}")
    print(f"Morts de Link : {deaths}")
    print(f"Limites de temps atteintes : {timeouts}")
    print(f"Récompense moyenne : {np.mean(rewards):.3f}")
    print(f"Actions moyennes : {np.mean(episode_lengths):.2f}")
    print(f"Attaques totales de l'ennemi : {total_enemy_attacks}")

    if successful_lengths:
        print(
            "Actions moyennes lors des victoires : "
            f"{np.mean(successful_lengths):.2f}"
        )

    print(f"Déplacements totaux de l'ennemi : {total_enemy_moves}")

    print(
        f"Déplacements moyens par épisode : "
        f"{total_enemy_moves / n_episodes:.2f}"
    )

    # ---------------------------------
    # Analyse des défaites
    # ---------------------------------

    print("\n=== Analyse des défaites ===")

    for result in death_details[:5]:

        print(f"\nÉpisode {result['episode']}")
        print(f"Link : {result['player_pos']}")
        print(f"Ennemi : {result['enemy_pos']}")
        print(f"PV Link : {result['player_hp']}")
        print(f"PV ennemi : {result['enemy_hp']}")
        print(f"Nombre d'actions : {result['steps']}")

        print("Répartition des actions :", result["action_counts"])
        print("Dernières actions :", result["last_actions"])

        print("\n20 derniers états :")

        for state in result["trace"]:
            print(state)

    # ---------------------------------
    # Analyse des timeouts
    # ---------------------------------

    print("\n=== Analyse des timeouts ===")

    for result in timeout_details[:5]:

        print(f"\nÉpisode {result['episode']}")
        print(f"Link : {result['player_pos']}")
        print(f"Ennemi : {result['enemy_pos']}")
        print(f"Distance : {result['distance']}")
        print(f"PV Link : {result['player_hp']}")
        print(f"PV ennemi : {result['enemy_hp']}")
        print(f"Nombre d'actions : {result['steps']}")

        print("Répartition des actions :", result["action_counts"])
        print("Dernières actions :", result["last_actions"])

        print("\n10 derniers états :")

        for state in result["trace"]:
            print(state)

    env.close()


if __name__ == "__main__":

    model = MaskablePPO.load(
        "models/maskable_ppo_aggressive_combat_v3_seed_123",
        device="cpu"
    )

    for seed in [2000, 3000, 4000, 5000, 6000]:

        print(f"\n===== ÉVALUATION SEED {seed} =====")

        evaluate_ppo(
            model,
            n_episodes=100,
            seed=seed,
            deterministic=False
        )