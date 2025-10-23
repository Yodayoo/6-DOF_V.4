import numpy as np

DH = np.array([
        [ 0.00,   np.pi/2,  1.00,  0.00    ],
        [ 2.50,   0.00,     0.00,  np.pi/2 ],
        [ 0.00,   np.pi/2,  0.00,  np.pi/2 ],
        [ 0.00,  -np.pi/2,  2.50,  0.00    ],
        [ 0.00,   np.pi/2,  0.00,  0.00    ],
        [ 0.00,   0.00,     0.50,  0.00    ],
    ])

q_min = None
q_max = None