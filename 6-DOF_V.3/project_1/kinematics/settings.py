import numpy as np

# DH Parameters: [a, alpha, d, theta0]
DH = np.array([
        [ 0.00,   np.pi/2,  1.00,  0.00    ],
        [ 2.50,   0.00,     0.00,  np.pi/2 ],
        [ 0.00,   np.pi/2,  0.00,  np.pi/2 ],
        [ 0.00,  -np.pi/2,  2.50,  0.00    ],
        [ 0.00,   np.pi/2,  0.00,  0.00    ],
        [ 0.00,   0.00,     0.50,  0.00    ],
    ])

# Workspace geometry
z_offset = np.hypot(DH[0 , 0] , DH[0 , 2])
max_r = sum(np.hypot(DH[1:4 , 0] , DH[1:4 , 2]))

# Joint limits (radians) - typical industrial robot limits
q_min = np.array([-np.pi, -np.pi/2, -np.pi, -np.pi, -np.pi/2, -np.pi])
q_max = np.array([np.pi, np.pi/2, np.pi, np.pi, np.pi/2, np.pi])

# Success thresholds for benchmarking
POS_ERROR_THRESHOLD = 1e-3   # 1 mm
ROT_ERROR_THRESHOLD = 1e-2   # ~0.57 degrees

# Number of joints
N_JOINTS = len(DH)