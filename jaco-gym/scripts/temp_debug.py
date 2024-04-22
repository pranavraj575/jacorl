import numpy as np

Pos1_coords = np.array([231, 169])
Pos2_coords = np.array([203, 84])
Pos3_coords = np.array([272, 92])

Pos1_trans = np.array([0.25211763, 0.14614685, 0.12006345])
Pos2_trans = np.array([0.26849404, -0.08071209, 0.11701352])
Pos3_trans = np.array([0.41985159, -0.04172455,  0.1462837])

Pos1_rot = np.array([[2.31931190e-02, -9.99730528e-01,  9.75493815e-04],
 [-1.23108865e-02,  6.90077760e-04,  9.99923980e-01],
 [-9.99655201e-01, -2.32033650e-02, -1.22915640e-02]])
print(np.cross(Pos1_rot[0], Pos1_rot[1]))
Pos2_rot = np.array([[-0.43709252, -0.7455187,   0.50314213],
 [ 0.47219343,  0.28590615,  0.83384113],
 [-0.76549558,  0.60204613,  0.22706159]])
Pos3_rot = np.array([[-0.41329611, -0.37260047,  0.83087617],
 [ 0.3499315,   0.77739979,  0.522683  ],
 [-0.84067489,  0.50677259, -0.19091167]])

intrinsic_matrix = np.array([[336.3354187,    0,         238.20429993],
 [0, 336.3354187, 128.42427063],
 [0, 0, 1]])


matrices = [(Pos1_coords, Pos1_rot, Pos2_trans), (Pos2_coords, Pos2_rot, Pos2_trans), (Pos3_coords, Pos3_rot, Pos3_trans)]
for coords, rot_matrix, translation_matrix in matrices:
    extrinsic = np.array([[rot_matrix[0, 0], rot_matrix[0, 1], rot_matrix[0, 2], translation_matrix[0]],
                                [rot_matrix[1, 0], rot_matrix[1, 1], rot_matrix[1, 2], translation_matrix[1]],
                                [rot_matrix[2, 0], rot_matrix[2, 1], rot_matrix[2, 2], translation_matrix[2]],
                                [0, 0, 0, 1]])
    
    thingy = np.array([rot_matrix[0], rot_matrix[1], translation_matrix]) / translation_matrix[2]
    print(thingy)
    

