import sys
import numpy as np

def read_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v '): # vertex position
                vertices.append(np.array(list(map(float, line.split()[1:]))))
            elif line.startswith('f '): # face indices
                faces.append(np.array(list(map(int, line.split()[1:]))))

    return np.array(vertices), np.array(faces)

def write_obj(vertices, faces, file_path):
    with open(file_path, 'w') as file: # write .obj file
        for v in vertices: # write vertex position
            file.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for f in faces: # write face indices
            file.write('f {} {} {}\n'.format(f[0], f[1], f[2]))

def laplacian_smoothing(vertices, faces, iterations=10, alpha=0.5):
    '''
    function: laplacian smoothing
    vertices: (num_vertices, 3) array of vertex positions
    faces: (num_faces, 3) array of face indices
    iterations: number of iterations
    alpha: step size
    returns: (num_vertices, 3) array of vertex positions'''
    vertex_count = len(vertices) # number of vertices
    adjacency_list = [[] for _ in range(vertex_count)] # adjacency list

    for f in faces:
        for i in range(3):
            adjacency_list[f[i] - 1].extend([f[(i + 1) % 3], f[(i + 2) % 3]])
            # add neighbors of vertex f[i] to adjacency list

    for _ in range(iterations):
        new_vertices = np.copy(vertices)
        for i in range(vertex_count):
            neighbors = np.unique(adjacency_list[i]) - 1
            # get neighbors of vertex i
            new_vertices[i] = vertices[i] * (1 - alpha) +\
                alpha * np.mean(vertices[neighbors], axis=0) # update vertex i
            
        vertices = new_vertices # update vertices

    return vertices

def main(input_obj, output_obj, iterations, alpha):
    '''
    function: main function
    input_obj: path to input .obj file
    output_obj: path to output .obj file
    iterations: number of iterations
    alpha: step size
    returns: None
    '''
    vertices, faces = read_obj(input_obj) # read .obj file
    smoothed_vertices = laplacian_smoothing(vertices, faces, iterations, alpha) 
    # laplacian smoothing
    write_obj(smoothed_vertices, faces, output_obj) # write .obj file

if __name__ == "__main__":
    input_obj = "data/smoothing.obj"
    output_obj = "output/new.obj"
    iterations = 10
    alpha = 0.7

    main(input_obj, output_obj, iterations, alpha)
