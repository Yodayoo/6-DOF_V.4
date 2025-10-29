import numpy as np

DH = np.array([
        [ 0.00,   np.pi/2,  1.00,  0.00    ],
        [ 2.50,   0.00,     0.00,  np.pi/2 ],
        [ 0.00,   np.pi/2,  0.00,  np.pi/2 ],
        [ 0.00,  -np.pi/2,  2.50,  0.00    ],
        [ 0.00,   np.pi/2,  0.00,  0.00    ],
        [ 0.00,   0.00,     0.50,  0.00    ],
    ])

z_offset = np.hypot(DH[0 , 0] , DH[0 , 2])
max_r = sum(np.hypot(DH[1:4 , 0] , DH[1:4 , 2]))

q_min = None
q_max = None