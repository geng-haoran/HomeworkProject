import numpy as np
import sys
import os
from scipy.sparse import csr_matrix, linalg

def read_obj(file_path):
    '''
    function: read .obj file
    file_path: path to input file
    returns: (vertices, faces)
    vertices: (num_vertices, 3) array of vertex positions
    faces: (num_faces, 3) array of face indices
    returns: None
    '''
    vertices = []
    faces = []

    with open(file_path, 'r') as f:
        lines = f.readlines() # read lines

    for line in lines:
        split_line = line.split()

        if not split_line:
            continue

        if split_line[0] == 'v':
            vertices.append([float(val) for val in split_line[1:]]) # vertex position
        elif split_line[0] == 'f':
            faces.append([int(val.split('/')[0]) - 1 for val in split_line[1:]]) 
            # face indices

    return np.array(vertices), np.array(faces)  # return vertices and faces

def write_obj(vertices, faces, file_path):
    '''
    function: write .obj file
    vertices: (num_vertices, 3) array of vertex positions
    faces: (num_faces, 3) array of face indices
    file_path: path to output file
    returns: None'''
    with open(file_path, 'w') as f:
        for vertex in vertices:
            f.write('v {} {} {}\n'.format(*vertex)) # write vertex position

        for face in faces:
            f.write('f {}\n'.format(' '.join([str(idx + 1) for idx in face]))) 
            # write face indices

def laplacian_deformation(vertices, faces, iterations=10, alpha=0.2):
    '''
    function: laplacian deformation
    vertices: (num_vertices, 3) array of vertex positions
    faces: (num_faces, 3) array of face indices
    iterations: number of iterations
    alpha: step size
    returns: (num_vertices, 3) array of vertex positions
    '''
    num_vertices = len(vertices)
    vertex_adjacency = {i: set() for i in range(num_vertices)}

    for face in faces:
        for i in range(3):
            vertex_adjacency[face[i]].add(face[(i + 1) % 3])
            vertex_adjacency[face[i]].add(face[(i + 2) % 3])

    for _ in range(iterations):
        new_vertices = np.copy(vertices)

        for vertex_id, neighbors in vertex_adjacency.items(): 
            avg_neighbor = np.mean(vertices[list(neighbors)], axis=0) # average neighbor
            laplacian = avg_neighbor - vertices[vertex_id] # laplacian
            new_vertices[vertex_id] += alpha * laplacian # update vertex position

        vertices = new_vertices

    return vertices

def arap_deformation(vertices, faces, anchor_vertex_ids, target_positions, 
                     iterations=10, local_iterations=10):
    '''
    function: arap deformation
    vertices: (num_vertices, 3) array of vertex positions
    faces: (num_faces, 3) array of face indices
    anchor_vertex_ids: (num_anchors,) array of vertex ids
    target_positions: (num_anchors, 3) array of target positions
    iterations: number of iterations
    local_iterations: number of local iterations
    returns: (num_vertices, 3) array of vertex positions
    '''
    num_vertices = len(vertices) # number of vertices
    vertex_adjacency = {i: set() for i in range(num_vertices)} # vertex adjacency

    for face in faces:
        for i in range(3):
            vertex_adjacency[face[i]].add(face[(i + 1) % 3])
            vertex_adjacency[face[i]].add(face[(i + 2) % 3])

    anchors = np.zeros((num_vertices, 3)) # anchor points
    anchors[anchor_vertex_ids] = target_positions

    weights = np.zeros((num_vertices, num_vertices)) # weight matrix

    for i in range(num_vertices):
        neighbors = list(vertex_adjacency[i])
        num_neighbors = len(neighbors) # number of neighbors

        for j, neighbor in enumerate(neighbors):
            weights[i, neighbor] = 1 / num_neighbors # weight matrix

    sparse_weights = csr_matrix(weights) # sparse weight matrix
    laplacian = csr_matrix(np.diag(np.sum(weights, axis=1))) - sparse_weights 
    # laplacian matrix

    for _ in range(iterations):
        local_rotations = []

        for i in range(num_vertices):
            neighbors = list(vertex_adjacency[i]) # neighbors
            rest_diff = vertices[neighbors] - vertices[i] # rest difference
            current_diff = vertices[neighbors] + anchors[neighbors] - anchors[i]

            covariance_matrix = current_diff.T @ rest_diff # covariance matrix
            u, _, vt = np.linalg.svd(covariance_matrix) # singular value decomposition

            rotation = u @ vt # rotation matrix
            local_rotations.append(rotation) # local rotations

        for _ in range(local_iterations):
            rhs = laplacian @ anchors # right hand side

            for i in range(num_vertices): # update right hand side
                rhs[i] += np.sum([weights[i, j] * local_rotations[j] 
                            @ (vertices[j] - vertices[i]) for j in vertex_adjacency[i]], axis=0)

            anchors = linalg.spsolve(laplacian + csr_matrix(np.diag(np.ones(num_vertices))), rhs) 
                # solve linear system

    deformed_vertices = vertices + anchors # deformed vertices
    return deformed_vertices

def build_laplacian(num_vertices, vertex_adjacency):
    '''
    function: build laplacian matrix
    num_vertices: number of vertices
    vertex_adjacency: vertex adjacency
    returns: laplacian matrix
    '''
    weights = np.zeros((num_vertices, num_vertices))

    for i in range(num_vertices):
        neighbors = list(vertex_adjacency[i])
        num_neighbors = len(neighbors)

        for j, neighbor in enumerate(neighbors):
            weights[i, neighbor] = 1 / num_neighbors

    sparse_weights = csr_matrix(weights)
    laplacian = csr_matrix(np.diag(np.sum(weights, axis=1))) - sparse_weights
    return laplacian

def poisson_deformation(vertices, faces, anchor_vertex_ids, target_positions, iterations=10):
    '''
    function: poisson deformation
    vertices: (num_vertices, 3) array of vertex positions
    faces: (num_faces, 3) array of face indices
    anchor_vertex_ids: (num_anchors,) array of vertex ids
    target_positions: (num_anchors, 3) array of target positions
    iterations: number of iterations
    '''
    num_vertices = len(vertices)
    vertex_adjacency = {i: set() for i in range(num_vertices)}

    for face in faces:
        for i in range(3):
            vertex_adjacency[face[i]].add(face[(i + 1) % 3])
            vertex_adjacency[face[i]].add(face[(i + 2) % 3])

    anchors = np.zeros((num_vertices, 3))
    anchors[anchor_vertex_ids] = target_positions

    laplacian = build_laplacian(num_vertices, vertex_adjacency)
    laplacian_coords = laplacian @ vertices

    for _ in range(iterations):
        anchors[anchor_vertex_ids] = target_positions
        deformed_vertices = linalg.spsolve(laplacian + csr_matrix(np.diag(np.ones(num_vertices))), laplacian_coords + anchors)
        vertices = np.reshape(deformed_vertices, (-1, 3))

    return vertices

def run_poisson(input_obj_file, output_obj_file):
    '''
    function: run poisson deformation
    input_obj_file: path to input obj file
    output_obj_file: path to output obj file
    returns: None
    '''
    vertices, faces = read_obj(input_obj_file)

    # Example anchor_vertex_ids and target_positions.
    # Replace these with your desired anchor vertices and target positions.
    anchor_vertex_ids = [0, 10]
    target_positions = np.array([[1, 0, 0], [-1, 0, 0]])

    deformed_vertices = poisson_deformation(vertices, faces, anchor_vertex_ids, target_positions)
    write_obj(deformed_vertices, faces, output_obj_file)

def run_laplacian(input_obj_file, output_obj_file):
    '''
    function: run laplacian deformation
    input_obj_file: path to input obj file
    output_obj_file: path to output obj file
    returns: None
    '''
    vertices, faces = read_obj(input_obj_file)
    deformed_vertices = laplacian_deformation(vertices, faces)
    write_obj(deformed_vertices, faces, output_obj_file)

def run_arap(input_obj_file, output_obj_file): 
    '''
    function: run arap deformation
    input_obj_file: path to input obj file
    output_obj_file: path to output obj file
    returns: None
    '''
    vertices, faces = read_obj(input_obj_file)

    # Example anchor_vertex_ids and target_positions.
    # Replace these with your desired anchor vertices and target positions.
    anchor_vertex_ids = [0, 10]
    target_positions = np.array([[1, 0, 0], [-1, 0, 0]])

    deformed_vertices = arap_deformation(vertices, faces, anchor_vertex_ids, target_positions)
    write_obj(deformed_vertices, faces, output_obj_file)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python deformation.py <method_code> <input_obj_file> <output_obj_file>")
        print("method_code: 1(poisson) 2(laplacian) 3(arap)")
        exit(1)

    method_code = sys.argv[1]
    input_obj_file = sys.argv[2]
    output_obj_file = sys.argv[3]

    if not os.path.isfile(input_obj_file):
        print("Error: Input OBJ file not found.")
        exit(1)
    if not os.path.isfile(output_obj_file):
        print("Output file can not find, generate one!")
        os.system("touch {}".format(output_obj_file))

    if method_code == "1":
        run_poisson(input_obj_file, output_obj_file)
    elif method_code == "2":
        run_laplacian(input_obj_file, output_obj_file)
    elif method_code == "3":
        run_arap(input_obj_file, output_obj_file)
