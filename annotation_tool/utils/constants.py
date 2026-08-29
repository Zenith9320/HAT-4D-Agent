import numpy as np

# Spherical harmonics DC component multiplier.
SH_C0 = 0.28209479177387814

# Keep viewer/world alignment consistent across frontend and backend.
# COORD_ALIGN_MAT = np.diag([-1.0, -1.0, -1.0]).astype(np.float32)
COORD_ALIGN_MAT = np.array([[0, 0, 1 ],
                            [1, 0, 0] ,
                            [0,1,0]    ]).astype(np.float32)



# COORD_ALIGN_MAT = np.diag([1.0, 1.0, 1.0]).astype(np.float32)

COORD_ALIGN_MAT_v2 = np.array([[-1, 0, 0 ],
                            [0, -1, 0] ,
                            [0,0,-1]    ]).astype(np.float32)