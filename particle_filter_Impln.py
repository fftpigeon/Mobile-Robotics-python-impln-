import numpy as np

from numpy.random import uniform
import matplotlib.pyplot as plt

import scipy as scipy


class filter_implementation():

    def __init__(self, r, w, velocity_l, velocity_r, std_velocity_l, std_velocity_r, variance_measurement):

        self.r = r
        self.w = w

        self.velocity_l = velocity_l
        self.velocity_r = velocity_r
        self.std_vel_l= std_velocity_l
        self.std_vel_r = std_velocity_r

        self.variance_measurement = variance_measurement

        self.velocity_l_change = []
        self.velocity_r_change = []
        self.num_of_particles = 1000

    def __particle_filter_propagate__(self, initial_points, time_step):
        #Find the new velocities
        std_new_vel = np.random.randn(self.num_of_particles) * self.std_vel_l
        self.velocity_l_change = self.velocity_l + std_new_vel
        std_new_vel = np.random.randn(self.num_of_particles) * self.std_vel_r
        self.velocity_r_change = self.velocity_r + std_new_vel

        accumulated_new_pose = np.zeros(shape=(3, self.num_of_particles), dtype=float)

        for index in range(self.num_of_particles):
            # Calculate initial pose
            x, y, angle = initial_points[0, index], initial_points[1, index], initial_points[2, index]

            x_pose_initial = np.matrix([[np.cos(angle), -np.sin(angle), x], [np.sin(angle), np.cos(angle), y], [0, 0, 1]])

            # Calculate the translated pose
            matrix_elem_a = (self.r / self.w) * (self.velocity_r_change[index] - self.velocity_l_change[index])
            matrix_elem_b = 0.5 * self.r * (self.velocity_r_change[index] + self.velocity_l_change[index])
            vel_map = np.matrix([[0, -matrix_elem_a, matrix_elem_b], [matrix_elem_a, 0, 0], [0, 0, 0]])

            translated_pose = x_pose_initial.dot(scipy.linalg.expm(time_step * vel_map))


            accumulated_new_pose[0, index], accumulated_new_pose[1, index], accumulated_new_pose[2, index] = \
                (translated_pose[0, 2], translated_pose[1, 2], np.arccos(translated_pose[0, 0]))
        return accumulated_new_pose

    def __particle_filter_update__(self, X1, Zt):
        X1 = np.array(X1)
        weights = np.ones(shape=(1, self.num_of_particles))

        for k in range(self.num_of_particles):
            # print(Zt)
            # print(X1)

            diff_matrix = Zt - np.matrix([[X1[0, k]], [X1[1, k]]])

            weights[0, k] = (1 / ((2 * np.pi)**0.5 * self.variance_measurement)) * scipy.linalg.expm(
                (-0.5 / self.variance_measurement**2) * (diff_matrix.T.dot(diff_matrix)))

        weights = weights / np.sum(weights)

        positions = (np.arange(self.num_of_particles) + np.random.random()) / self.num_of_particles

        indexes = np.zeros(self.num_of_particles, int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < self.num_of_particles and j < self.num_of_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1

        X1[0, :] = X1[0, indexes]
        X1[1, :] = X1[1, indexes]
        X1[2, :] = X1[2, indexes]
        # print(X1)
        return (X1)




particle_filter = filter_implementation(r=0.25, w=0.5, velocity_l=1.5, velocity_r=2,
                                        std_velocity_l=0.05, std_velocity_r=0.05, variance_measurement=0.1)

## Plot point clouds
fig = plt.figure()
ax = plt.axes()

particles_new = particle_filter.__particle_filter_propagate__(np.zeros(shape=(3, particle_filter.num_of_particles)), 5)
ax.scatter(particles_new[0], particles_new[1])
particles_new = particle_filter.__particle_filter_update__(particles_new, np.matrix([[1.6561], [1.2847]]))
ax.scatter(particles_new[0], particles_new[1])
print("Mean and covariance after first update")
print(particles_new.mean(), np.cov(particles_new))

# particles_new = particle_filter.__particle_filter_propagate__(np.zeros(shape=(3, particle_filter.num_of_particles)), 10)
# ax.scatter(particles_new[0], particles_new[1])
# particles_new = particle_filter.__particle_filter_update__(particles_new, np.matrix([[1.0505], [3.1059]]))
# ax.scatter(particles_new[0], particles_new[1])
# print("Mean and covariance after second update")
# print(particles_new.mean(), np.cov(particles_new))
# #
# particles_new = particle_filter.__particle_filter_propagate__(np.zeros(shape=(3, particle_filter.num_of_particles)), 15)
# ax.scatter(particles_new[0], particles_new[1])
# particles_new = particle_filter.__particle_filter_update__(particles_new, np.matrix([[-0.9875], [3.2118]]))
# ax.scatter(particles_new[0], particles_new[1])
# print("Mean and covariance after third update")
# print(particles_new.mean(), np.cov(particles_new))
# #
# particles_new = particle_filter.__particle_filter_propagate__(np.zeros(shape=(3, particle_filter.num_of_particles)), 20)
# ax.scatter(particles_new[0], particles_new[1])
# particles_new = particle_filter.__particle_filter_update__(particles_new, np.matrix([[-1.6450], [1.1978]]))
# ax.scatter(particles_new[0], particles_new[1])
# print("Mean and covariance after fourth update")
# print(particles_new.mean(), np.cov(particles_new))

plt.show()
