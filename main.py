import numpy as np
import matplotlib.pyplot as plt


sun = {"position": np.array([5, 5, 5]), "ambient": np.array([1, 1, 1]), "diffuse": np.array([1, 1, 1]),
       "specular": np.array([1, 1, 1])}
spheres = [
    {"center": np.array([0.1, -0.3, -1]), "radius": 0.1, "ambient": np.array([0.1, 0, 0]),
     "diffuse": np.array([0.8, 0, 0]), "specular": np.array([1, 1, 1]), "shininess": 100},
    {"center": np.array([-0.5, 0.6, -2]), "radius": 0.5, "ambient": np.array([0, 0.1, 0.1]),
     "diffuse": np.array([0, 0.8, 0.8]), "specular": np.array([1, 1, 1]), "shininess": 100},
    {"center": np.array([-0.8, 0, -0.5]), "radius": 0.2, "ambient": np.array([0.1, 0.1, 0]),
     "diffuse": np.array([0.8, 0.8, 0]), "specular": np.array([1, 1, 1]), "shininess": 100},
    {"center": np.array([-0.6, 0.8, -0.5]), "radius": 0.1, "ambient": np.array([0, 0.1, 0]),
     "diffuse": np.array([0, 0.8, 0]), "specular": np.array([1, 1, 1]), "shininess": 100}
]


def normalize(vector):
    return vector/np.linalg.norm(vector)


def sphere_intersection_distance(center, radius, start, ray):
    b = 2*np.dot(ray, start - center)
    c = np.linalg.norm(start - center)**2 - radius**2
    discriminant = b**2 - 4*c
    if discriminant > 0:
        sol1 = (-b + np.sqrt(discriminant))/2
        sol2 = (-b - np.sqrt(discriminant))/2
        nearest_distance = min(sol1, sol2)
        if nearest_distance > 0:
            return nearest_distance
    return np.inf


def nearest_sphere_intersection(spheres, start, ray):
    distances = [sphere_intersection_distance(sphere["center"], sphere["radius"], start, ray) for sphere in spheres]
    closest_sphere = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance < min_distance:
            min_distance = distance
            closest_sphere = spheres[index]
    return closest_sphere, min_distance


camera = np.array([0, 0, 1])
screen = (-1, 1, 1, -1)

image = np.zeros((500, 500, 3))

for i, y in enumerate(np.linspace(1, -1, 500)):
    for j, x in enumerate(np.linspace(-1, 1, 500)):
        pixel = np.array([x, y, 0])
        ray = normalize(pixel - camera)
        start = camera
        closest_sphere, min_distance = nearest_sphere_intersection(spheres, start, ray)
        if closest_sphere is None:
            continue
        sphere_intersection = start + ray * min_distance
        radius_vector = normalize(sphere_intersection - closest_sphere["center"])
        offset_point = sphere_intersection + (10**(-5))*radius_vector
        ray_to_sun = normalize(sun["position"] - offset_point)
        blocker, min_distance = nearest_sphere_intersection(spheres, offset_point, ray_to_sun)
        sphere_to_sun_distance = np.linalg.norm(sun["position"] - offset_point)
        if min_distance < sphere_to_sun_distance:
            continue
        color = np.zeros(3)
        color = color + closest_sphere["ambient"]*sun["ambient"]
        color = color + closest_sphere["diffuse"]*sun["diffuse"]*np.dot(ray_to_sun, radius_vector)
        ray_to_camera = -1*ray
        halfway = normalize(ray_to_camera + ray_to_sun)
        color = color + closest_sphere["specular"]*sun["specular"] *\
            (np.dot(radius_vector, halfway)**(closest_sphere["shininess"]/4))
        image[i, j] = np.clip(color, 0, 1)

plt.imshow(image)
plt.show()




