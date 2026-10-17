import numpy as np

from src.environment.zelda_env import ZeldaEnv


def observation_to_state(observation, grid_size):
    """
    Convertit l'observation normalisée en état discret.
    Necessaire pour que le q agent puisse interpreter les données

    Observation :
        [player_x, player_y, target_x, target_y]

    Les coordonnées sont converties en entiers.
    """
    coordinates = np.rint(
        observation * (grid_size - 1)
    ).astype(int)

    coordinates = np.clip(
        coordinates, 0, grid_size - 1
    )

    return tuple(coordinates)


def train_q_learning(
    env,
    n_episodes=10_000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9995,
    seed=42,
):
    """
    Entraîne un agent avec Q-learning tabulaire.

    alpha : taux d'apprentissage
    gamma : importance des récompenses futures
    epsilon : probabilité d'explorer au lieu d'exploiter
    """

    rng = np.random.default_rng(seed)
    grid_size = env.grid_size
    n_actions = env.action_space.n

    # Q[x_joueur, y_joueur, x_cible, y_cible, action]
    q_table = np.zeros(
        (
            grid_size,
            grid_size,
            grid_size,
            grid_size,
            n_actions,
        ),
        dtype=np.float32,
    )

    epsilon = epsilon_start

    history = {
        "rewards": [],
        "steps": [],
        "successes": [],
    }

    for episode in range(n_episodes):

        # Utiliser une graine différente par épisode,
        # tout en permettant de reproduire l'entraînement.
        observation, info = env.reset(seed=seed + episode)
        state = observation_to_state(
            observation, grid_size
        )

        total_reward = 0.0
        steps = 0

        while True:

            # 1. Choix de l'action : exploration ou exploitation.
            if rng.random() < epsilon:
                # Exploration : choisir une action au hasard.
                action = int(rng.integers(n_actions))
            else:
                # Exploitation : choisir une meilleure action connue.
                q_values = q_table[state]

                # Si plusieurs actions sont à égalité,
                # en choisir une aléatoirement.
                best_actions = np.flatnonzero(
                    q_values == np.max(q_values)
                )

                action = int(rng.choice(best_actions))

            # 2. Exécuter l'action dans Gymnasium.
            next_observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            next_state = observation_to_state(
                next_observation, grid_size
            )

            # 3. Calculer la cible de mise à jour Q-learning.
            if terminated:
                target = reward
            else:
                target = reward + gamma * np.max(
                    q_table[next_state]
                )

            # 4. Mise à jour de Q(s, a).
            old_value = q_table[state + (action,)]

            q_table[state + (action,)] = (
                old_value
                + alpha * (target - old_value)
            )

            total_reward += reward
            steps += 1

            # 5. Passer au nouvel état.
            state = next_state

            # Terminé si la cible est atteinte ou si
            # la limite de temps de l'épisode est atteinte.
            if terminated or truncated:
                break

        # 6. Enregistrer les résultats de l'épisode.
        history["rewards"].append(total_reward)
        history["steps"].append(steps)
        history["successes"].append(int(terminated))

        # Réduire progressivement l'exploration.
        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay,
        )

        if (episode + 1) % 1_000 == 0:
            last_successes = history["successes"][-100:]
            last_rewards = history["rewards"][-100:]

            print(
                f"Épisode {episode + 1:>5}/{n_episodes} | "
                f"Epsilon : {epsilon:.3f} | "
                f"Réussite (100 derniers) : "
                f"{np.mean(last_successes):.1%} | "
                f"Récompense moyenne : "
                f"{np.mean(last_rewards):.3f}"
            )

    return q_table, history


def play_episode(env, q_table, seed=None, render=False):
    """
    Fait jouer l'agent entraîné sans exploration volontaire.
    Il choisit une meilleure action connue dans la Q-table.
    """

    rng = np.random.default_rng(seed)

    observation, info = env.reset(seed=seed)
    state = observation_to_state(
        observation, env.grid_size
    )

    total_reward = 0.0
    steps = 0

    if render:
        print("État initial :")
        env.render()

    while True:

        # Choisir une action qui maximise Q(s, a).
        q_values = q_table[state]
        best_actions = np.flatnonzero(
            q_values == np.max(q_values)
        )
        action = int(rng.choice(best_actions))

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        state = observation_to_state(
            observation, env.grid_size
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

        if terminated or truncated:
            break

    return total_reward, steps, terminated


def evaluate_q_agent(env, q_table, n_episodes=100, seed=10_000):
    """Évalue l'agent entraîné sur plusieurs épisodes."""

    rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(n_episodes):
        reward, steps, terminated = play_episode(
            env,
            q_table,
            seed=seed + episode,
        )

        rewards.append(reward)
        episode_lengths.append(steps)
        successes += int(terminated)

    print("\n=== Résultats de l'agent Q-learning ===")
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

    env = ZeldaEnv(
        grid_size=5,
        max_steps=50,
    )

    q_table, history = train_q_learning(
        env,
        n_episodes=10_000,
    )

    evaluate_q_agent(
        env,
        q_table,
        n_episodes=100,
    )

    print("\n=== Démonstration de l'agent entraîné ===")

    demo_env = ZeldaEnv(
        grid_size=5,
        max_steps=50,
        render_mode="human",
    )

    play_episode(
        demo_env,
        q_table,
        seed=123,
        render=True,
    )

    env.close()
    demo_env.close()