# field_ring_optimizer/constants.py

# --- Physical Constants ---
UM_TO_CM = 1e-4  # Micrometer to Centimeter
V_PER_UM_TO_MV_PER_CM = 1e-2  # V/um to MV/cm conversion
L_REF = 10.0  # Reference length for non-dimensionalization (μm)
V_REF = 2000.0  # Reference voltage (V)
EPSILON_R = 9.0  # Relative permittivity of the material
EPSILON_0 = 8.85e-14  # Permittivity of free space (F/cm)
RHO_INSIDE_C_PER_CM3 = 1e18 * 1.602e-19  # Charge density inside (C/cm^3)

# --- Non-dimensional Poisson Parameter ---
# This is the term f in ∇²u = -f
POISSON_PARAM = ((L_REF * UM_TO_CM)**2 * RHO_INSIDE_C_PER_CM3) / (EPSILON_R * EPSILON_0 * V_REF)

# --- Optimization Constants ---
TARGET_FIELD_MV_PER_CM = 4.0  # Target electric field (MV/cm)
MIN_GEOMETRY_VALUE = 1.0      # Minimum allowed value for s and w parameters (μm)