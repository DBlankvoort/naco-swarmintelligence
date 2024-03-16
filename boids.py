import random
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import json


class SimParams:
    """
    Abstracts away a simulation to its base parameters
    """

    def __init__(
        self,
        num_dims=2,  # number of dimensions of the space (int)
        dim_width=np.array([600, 600]),  # width of each dimension (ndarray)
        num_steps=300,  # number of steps the simulation runs (int)
        speed=5,  # speed of movement (int)
        radius=100,  # radius of neighborhood (float)
        max_angle=np.pi * 0.75,  # 'angle of vision' of neighborhood (float)
        cohesion=0.05,  # cohesion force modifier (float)
        separation=0.05,  # separation force modifier (float)
        alignment_strength=0.05,  # alignment strength force modifier (float)
        num_boids=15,
    ):  # number of boids in the simulation (int)
        self.num_dims = num_dims
        self.dim_width = dim_width
        self.num_steps = num_steps
        self.speed = speed
        self.cohesion = cohesion
        self.separation = separation
        self.radius = radius
        self.max_angle = max_angle
        self.alignment_strength = alignment_strength
        self.num_boids = num_boids


class Boid:
    """
    A boid is equipped with a position and a velocity vector,
    as well as with a function `neighbors' which gets the
    subset of boids which are its neighbors and a function
    `update' which performs one time step of calculations
    for the boid.
    """

    def __init__(self, pos, v):
        self.pos = pos
        self.v = v

    def neighbors(self, params, boids):
        """
        Gets the neighbors of the boid.
        """

        neighs = []

        for boid in boids:
            if np.any(boid.pos != self.pos):

                # Ensure behavior at the boundaries is handled correctly
                boid_modified = Boid(
                    np.array([boid.pos[i] for i in range(params.num_dims)]), boid.v
                )
                for i in range(params.num_dims):
                    if (
                        self.pos[i] + (params.dim_width[i] - boid.pos[i])
                        < params.radius
                    ):
                        boid_modified.pos[i] -= params.dim_width[i]
                    if (params.dim_width[i] - self.pos[i]) + boid.pos[
                        i
                    ] < params.radius:
                        boid_modified.pos[i] += params.dim_width[i]

                dist = np.linalg.norm(boid_modified.pos - self.pos)

                ang = np.arccos(
                    np.clip(
                        np.dot(
                            (boid_modified.pos - self.pos)
                            / np.linalg.norm(boid_modified.pos - self.pos),
                            self.v / np.linalg.norm(self.v),
                        ),
                        -1.0,
                        1.0,
                    )
                )
                if dist < params.radius and abs(ang) < params.max_angle:
                    neighs.append(boid_modified)
        return neighs

    def update(self, params, dim_width, boids):
        """
        Updates the positions and velocities
        of the particle according to the Boid
        algorithm.
        """

        neighs = self.neighbors(params, boids)

        nn_distance = math.inf
        for boid in boids:
            if np.any(boid.pos != self.pos):
                boid_modified = Boid(
                    np.array([boid.pos[i] for i in range(params.num_dims)]), boid.v
                )
                for i in range(params.num_dims):
                    if (
                        self.pos[i] + (params.dim_width[i] - boid.pos[i])
                        < params.radius
                    ):
                        boid_modified.pos[i] -= params.dim_width[i]
                    if (params.dim_width[i] - self.pos[i]) + boid.pos[
                        i
                    ] < params.radius:
                        boid_modified.pos[i] += params.dim_width[i]
                dist = np.linalg.norm(boid_modified.pos - self.pos)
                nn_distance = min(nn_distance, dist)

        if len(neighs) > 0:
            avg_v = 0
            avg_pos = 0
            for neigh in neighs:
                avg_v += neigh.v
                avg_pos += neigh.pos

            avg_pos /= len(neighs)
            avg_v /= len(neighs)

            # Alignment
            self.v += params.alignment_strength * avg_v

            # Cohesion
            self.v += params.cohesion * (avg_pos - self.pos)

            # Separation
            self.v -= params.separation * (avg_pos - self.pos)

        # Renormalize velocity vectors
        self.v = self.v / np.linalg.norm(self.v)

        self.pos = (self.pos + params.speed * self.v) % dim_width

        return nn_distance


def update_frame(frame, scatter, boids, order_text, step_text, hist_ax):

    nn_distances = []
    # Update the positions of boids
    for boid in boids:
        nn_distances.append(boid.update(params, params.dim_width, boids))

    # Update scatter plot data
    scatter.set_offsets([[boid.pos[0], boid.pos[1]] for boid in boids])

    # Calculate and display order
    order = (1 / params.num_boids) * np.linalg.norm(
        sum([boid.v / np.linalg.norm(boid.v) for boid in boids])
    )
    order_text.set_text(f"Order: {order}")
    step_text.set_text(f"Number of steps: {frame}")

    # Plot histogram of nearest neighbor distances
    hist_ax.clear()
    hist_ax.hist(
        nn_distances, bins=np.arange(0, 601, 25), color="skyblue", edgecolor="black"
    )
    hist_ax.set_xlim(0, 300)
    hist_ax.set_ylim(0, len(boids))
    hist_ax.set_title("Nearest Neighbor Distances")
    hist_ax.set_xlabel("Distance")
    hist_ax.set_ylabel("Frequency")


def run_simulation(params, show_anims=True):
    boids = [
        Boid(
            np.array(
                [random.uniform(0, params.dim_width[i]) for i in range(params.num_dims)]
            ),
            np.array([random.uniform(-0.5, 0.5) for _ in range(params.num_dims)]),
        )
        for _ in range(params.num_boids)
    ]

    if show_anims:
        fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(10, 5))
        ax_scatter.set_title("Current boid positions")
        scatter = ax_scatter.scatter(
            [boid.pos[0] for boid in boids], [boid.pos[1] for boid in boids]
        )
        order_text = ax_scatter.text(0.02, 0.95, "", transform=ax_scatter.transAxes)
        step_text = ax_scatter.text(0.02, 0.90, "", transform=ax_scatter.transAxes)

        ani = FuncAnimation(
            fig,
            update_frame,
            frames=params.num_steps,
            fargs=(scatter, boids, order_text, step_text, ax_hist),
            interval=1,
        )
        plt.xlim([0, 600])
        plt.ylim([0, 600])
        plt.show()
    else:
        for i in range(params.num_steps):
            nn_dists = []
            for boid in boids:
                nn_dists.append(boid.update(params, params.dim_width, boids))

        order = (1 / params.num_boids) * np.linalg.norm(
            sum([boid.v / np.linalg.norm(boid.v) for boid in boids])
        )

        # plt.plot()
        # plt.hist(nn_dists)
        # plt.show()

        return order


def run_ABC(N, thresholds, cov):

    populations = [[]]

    # Get to N elements of the population
    print(f"Initial iteration (threshold={thresholds[0]})")
    while len(populations[0]) < N:

        cand = {
            "cohesion": random.uniform(0, 1),
            "separation": random.uniform(0, 1),
            "alignment_strength": random.uniform(0, 1),
        }
        performance = run_simulation(
            SimParams(
                cohesion=cand["cohesion"],
                separation=cand["separation"],
                alignment_strength=cand["alignment_strength"],
            ),
            show_anims=False,
        )

        cand["score"] = performance

        if performance > thresholds[0]:
            print(f"...Got a new member with order: {performance}")
            populations[0].append(cand)
        else:
            print(f"...Failed a candidate with order: {performance}")
        print(f"Current members: {len(populations[0])}/{N}")

    # Start selecting
    for i, threshold in enumerate(thresholds[1:]):

        print(f"Current iteration: {i+1} (threshold={threshold})")

        new_population = []

        while len(new_population) < N:

            cand = deepcopy(random.choice(populations[-1]))
            cand["cohesion"] += random.gauss(0, cov)
            cand["separation"] += random.gauss(0, cov)
            cand["alignment_strength"] += random.gauss(0, cov)

            performance = run_simulation(
                SimParams(
                    cohesion=cand["cohesion"],
                    separation=cand["separation"],
                    alignment_strength=cand["alignment_strength"],
                ),
                show_anims=False,
            )

            cand["score"] = performance

            if performance > threshold:
                new_population.append(cand)
                print(f"...Got a new member with order: {performance}")
            else:
                print(f"...Failed a candidate with order: {performance}")
            print(f"Current members: {len(new_population)}/{N}")

        populations.append(new_population)

    return populations


populations = run_ABC(20, list(np.arange(0.5, 1, 0.05)) + [0.99], 0.005)
with open("results.json", "w") as f:
    json.dump(populations, f)

# params = SimParams(num_dims=2,
#                    dim_width=np.array([600, 600]),
#                    num_steps=300,
#                    speed=5,
#                    radius=100,
#                    max_angle=np.pi / 0.75,
#                    cohesion=0.05,
#                    separation=0.05,
#                    alignment_strength=0.05,
#                    num_boids=15)


# print(run_simulation(params, show_anims=True))
