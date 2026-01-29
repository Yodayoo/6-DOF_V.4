# Inverse Kinematics Solver Comparison: Methodology Framework

## 1. Robot Model

### 1.1 Denavit-Hartenberg (DH) Parameters

The robot is modeled as a 6-DOF serial manipulator using the standard Denavit-Hartenberg convention. Each joint is described by four parameters:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Link length | a | Distance along x-axis from z_{i-1} to z_i |
| Link twist | α | Angle about x-axis from z_{i-1} to z_i |
| Link offset | d | Distance along z-axis from x_{i-1} to x_i |
| Joint angle | θ | Angle about z-axis from x_{i-1} to x_i |

**DH Parameter Table (from `settings.py`):**

| Joint | a (m) | α (rad) | d (m) | θ₀ (rad) |
|-------|-------|---------|-------|----------|
| 1 | 0.00 | π/2 | 1.00 | 0 |
| 2 | 2.50 | 0 | 0.00 | π/2 |
| 3 | 0.00 | π/2 | 0.00 | π/2 |
| 4 | 0.00 | -π/2 | 2.50 | 0 |
| 5 | 0.00 | π/2 | 0.00 | 0 |
| 6 | 0.00 | 0 | 0.50 | 0 |

### 1.2 Forward Kinematics

The forward kinematics computes the end-effector pose from joint angles using homogeneous transformation matrices.

**Individual Joint Transformation:**
```
T_i = Rot_z(θ_i) × Trans_z(d_i) × Trans_x(a_i) × Rot_x(α_i)
```

**In matrix form:**
```
T_i = | cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)   a·cos(θ) |
      | sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)   a·sin(θ) |
      |   0        sin(α)          cos(α)          d     |
      |   0          0               0             1     |
```

**Total End-Effector Transformation:**
```
T_total = T_1 × T_2 × T_3 × T_4 × T_5 × T_6
```

The resulting 4×4 matrix contains:
- Rotation matrix R (3×3): Orientation of end-effector
- Position vector p (3×1): Position of end-effector
- `T = [R | p; 0 0 0 | 1]`

### 1.3 Joint Limits

Physical joint constraints are defined as:

| Joint | Minimum (rad) | Maximum (rad) |
|-------|---------------|---------------|
| 1 | -π | +π |
| 2 | -π/2 | +π/2 |
| 3 | -π | +π |
| 4 | -π | +π |
| 5 | -π/2 | +π/2 |
| 6 | -π | +π |

---

## 2. Inverse Kinematics Solvers

### 2.1 Problem Definition

Given a target pose T_target (4×4 homogeneous transformation matrix), find joint angles θ = [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆] such that:

```
FK(θ) = T_target
```

### 2.2 Analytical Solver

**Algorithm Type:** Closed-form geometric solution

**Approach:**
1. **Position Decoupling:** Separates the problem into:
   - Arm position (joints 1-3): Determines wrist center position
   - Wrist orientation (joints 4-6): Determines end-effector orientation

2. **Wrist Center Calculation:**
   ```
   P_wrist = P_target - d_6 × R_target × [0, 0, 1]ᵀ
   ```
   Where d_6 is the tool offset (0.5m).

3. **Joint 1 Solution:**
   ```
   θ₁ = atan2(P_wrist_y, P_wrist_x)
   ```

4. **Joints 2-3 Solution:** Uses geometric relationships and the law of cosines:
   ```
   r = √(P_wrist_x² + P_wrist_y²)
   s = P_wrist_z - d_1
   D = (r² + s² - a_2² - d_4²) / (2 × a_2 × d_4)
   θ₃ = atan2(±√(1-D²), D)
   θ₂ = atan2(s, r) - atan2(d_4×sin(θ₃), a_2 + d_4×cos(θ₃))
   ```

5. **Joints 4-6 Solution:** Euler angle extraction from rotation matrix:
   ```
   R_0_3 = T_1 × T_2 × T_3  (rotation part)
   R_3_6 = R_0_3ᵀ × R_target
   θ₄, θ₅, θ₆ = Extract ZYZ Euler angles from R_3_6
   ```

**Characteristics:**
- Deterministic output
- Machine-precision accuracy (~10⁻¹⁵ m position error)
- Fastest computation (~0.3 ms per solve)
- May produce solutions outside joint limits

### 2.3 Numerical Solver

**Algorithm Type:** Damped Least Squares (DLS) iterative method with multi-restart

**Core Algorithm - Damped Least Squares:**

1. **Pose Error Computation:**
   ```
   e = [e_position; e_orientation]

   e_position = P_target - P_current  (3×1 vector)
   e_orientation = 0.5 × (R_current × R_targetᵀ - R_target × R_currentᵀ)_vee  (3×1 vector)
   ```
   Where (·)_vee extracts the vector from a skew-symmetric matrix.

2. **Jacobian Matrix (6×6):**
   The geometric Jacobian J relates joint velocities to end-effector velocities:
   ```
   [v; ω] = J × θ̇
   ```

   For each joint i:
   ```
   J_i = [z_{i-1} × (P_end - P_{i-1}); z_{i-1}]  (for revolute joints)
   ```
   Where z_{i-1} is the rotation axis of joint i.

3. **DLS Step Computation:**
   ```
   Δθ = Jᵀ × (J × Jᵀ + λ²I)⁻¹ × e
   ```
   Where λ is the damping factor.

4. **Adaptive Damping:**
   - Initial: λ = 0.01
   - If error decreases: λ = max(λ_min, λ × 0.5), increase step size
   - If error increases: λ = min(λ_max, λ × 2.0), decrease step size
   - λ_min = 10⁻⁶, λ_max = 10

5. **Convergence Criteria:**
   ```
   ||e|| < tolerance  (default: 10⁻⁶)
   ```

6. **Multi-Restart Strategy:**
   - Attempts from multiple initial configurations
   - Initial guesses: [user-provided, zero configuration, random within limits]
   - Returns best solution found (lowest error)

**Stall Detection:**
- Tracks improvement over iterations
- Terminates early if no improvement for 20 consecutive iterations

**Characteristics:**
- Iterative refinement
- Typical accuracy: ~10⁻⁷ m position error
- Slower computation (~8 ms per solve)
- Better joint limit awareness
- 99.96% convergence rate

---

## 3. Benchmark Methodology

### 3.1 Target Generation

Targets are generated as valid 4×4 homogeneous transformation matrices within the robot's reachable workspace:

1. **Position Sampling:**
   - Random positions within workspace sphere
   - Filtered to ensure reachability

2. **Orientation Sampling:**
   - Random rotations uniformly distributed on SO(3)
   - Ensures diverse end-effector orientations

### 3.2 Batch Processing

```
For each batch of N targets:
    1. Record start time and CPU state
    2. For each target in batch:
        a. Solve IK
        b. Compute FK of solution
        c. Measure pose error
        d. Record metrics
    3. Record end time and resource usage
    4. Compute batch statistics
```

### 3.3 Measured Parameters

#### 3.3.1 Accuracy Metrics

**Position Error:**
```
e_pos = ||P_target - P_solved||₂  (Euclidean distance in meters)
```

**Rotation Error:**
```
e_rot = angle(R_target × R_solvedᵀ)  (in radians)
```
Computed using scipy's Rotation.magnitude() which gives the angle of the rotation needed to align the two orientations.

#### 3.3.2 Success Criteria

A solution is considered **successful** if:
```
e_pos < 1 mm  AND  e_rot < 0.01 rad (~0.57°)
```

#### 3.3.3 Timing Metrics

- **Per-solve time:** `time.time()` difference for each solve
- **Total time:** Sum of all solve times
- **Throughput:** `N_solves / total_time` (solves per second)

#### 3.3.4 Joint Limit Analysis

**Within Limits Check:**
```
within_limits = all(q_min ≤ θ ≤ q_max)
```

**Joint Margin:** Minimum distance to any joint limit
```
margin = min(min(θ - q_min), min(q_max - θ))
```
Positive margin means within limits; negative means violation.

#### 3.3.5 Manipulability Index

**Yoshikawa Manipulability:**
```
w = √(det(J × Jᵀ))
```

Physical interpretation:
- Higher values indicate configurations far from singularities
- Zero at singular configurations
- Measures the "quality" of a configuration for motion in all directions

#### 3.3.6 Numerical Solver Specific

- **Iterations:** Total iterations across all restarts
- **Convergence:** Boolean - did solver reach tolerance?
- **Restarts:** Number of initial guesses attempted

#### 3.3.7 Workspace Analysis

Targets are categorized by position:

**Radial Distance (from base):**
- Inner: r < 1.5 m
- Middle: 1.5 m ≤ r < 3.5 m
- Outer: r ≥ 3.5 m

**Height (z-coordinate):**
- Low: z < 0.5 m
- Mid: 0.5 m ≤ z < 2.0 m
- High: z ≥ 2.0 m

Regions: `{inner, middle, outer} × {low, mid, high}` = 9 regions

#### 3.3.8 Resource Usage

- **Memory:** Process memory delta per batch (MB)
- **CPU Utilization:** `(cpu_user + cpu_system) / wall_time × 100%`
- **CPU Frequency:** Current processor frequency during benchmark

#### 3.3.9 Computing Power Estimates

**FLOPS Estimation:**
```
Analytical: ~500 FLOPs per solve (direct computation)
Numerical: ~5000 FLOPs × iterations per solve (Jacobian + DLS)
```

**Throughput-based MFLOPS:**
```
MFLOPS = (total_FLOPs / total_time) / 10⁶
```

---

## 4. Statistical Analysis

### 4.1 Descriptive Statistics

For each metric, we compute:
- **Mean (μ):** Average value
- **Standard Deviation (σ):** Spread of values
- **Median:** 50th percentile (robust to outliers)
- **Percentiles:** 25th, 75th, 95th, 99th
- **Range:** Min, Max
- **IQR:** Interquartile range (P75 - P25)

### 4.2 Statistical Tests

#### 4.2.1 Welch's t-test

Tests whether means of two groups are significantly different.

**Assumptions:**
- Samples are independent
- Approximately normally distributed (or large sample size)
- Does NOT assume equal variances

**Test Statistic:**
```
t = (μ_A - μ_N) / √(s_A²/n_A + s_N²/n_N)
```

**Interpretation:**
- p < 0.05: Statistically significant difference
- p < 0.01: Highly significant
- p < 0.001: Very highly significant

#### 4.2.2 Mann-Whitney U Test

Non-parametric alternative to t-test.

**Advantages:**
- No normality assumption
- Robust to outliers
- Tests if one distribution is stochastically greater

**Use Case:** When data is not normally distributed (e.g., timing data often has long tails)

#### 4.2.3 Cohen's d Effect Size

Measures the practical significance of the difference.

**Formula:**
```
d = (μ₁ - μ₂) / s_pooled

s_pooled = √(((n₁-1)s₁² + (n₂-1)s₂²) / (n₁ + n₂ - 2))
```

**Interpretation:**
| |d| | Effect Size |
|-----|-------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

### 4.3 Correlation Analysis

**Pearson Correlation Coefficient (r):**
```
r = Σ((x_i - μ_x)(y_i - μ_y)) / (n × σ_x × σ_y)
```

**Interpretation:**
| |r| | Correlation Strength |
|-----|---------------------|
| < 0.1 | Negligible |
| 0.1 - 0.3 | Weak |
| 0.3 - 0.5 | Moderate |
| > 0.5 | Strong |

Used to analyze:
- Distance from base vs. position error
- Target height vs. position error

---

## 5. Code Architecture

### 5.1 Directory Structure

```
project_1/
├── kinematics/
│   ├── settings.py          # DH parameters, joint limits, thresholds
│   ├── FK_chain.py           # Forward kinematics implementation
│   ├── analytical_solver.py  # Closed-form IK solver
│   └── numerical_solver.py   # Iterative DLS solver
├── benchmark/
│   ├── bench_func.py         # Benchmark framework and metrics
│   ├── benchmark_analytical.py
│   └── benchmark_numerical.py
├── analysis/
│   ├── solver_comparison.py  # Statistical comparison report
│   └── plot_comparison.py    # Visualization generation
├── target_gen/
│   └── generate_targets.py   # Target pose generation
└── data/
    ├── target_archive.npy    # Pre-generated targets
    ├── analytical_solutions/ # Benchmark results
    ├── numerical_solutions/
    └── plots/                # Generated figures
```

### 5.2 Data Flow

```
┌─────────────────┐
│  Target         │
│  Generation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Target Archive │
│  (1M poses)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Analyt.│ │Numer. │
│Solver │ │Solver │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│Results│ │Results│
│ .npy  │ │ .npy  │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│  Statistical    │
│  Comparison     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Report & Plots │
└─────────────────┘
```

### 5.3 Key Functions

**`measure_batch_performance()`** - Main benchmark function
- Inputs: solver function, target array, batch size, solver type
- Outputs: Dictionary containing all metrics

**`compute_error()`** - Accuracy measurement
- Inputs: solved angles, target transformation
- Outputs: position error (m), rotation error (rad)

**`compute_manipulability()`** - Configuration quality
- Inputs: DH parameters, joint angles
- Outputs: Yoshikawa index (scalar)

**`check_joint_limits()`** - Constraint validation
- Inputs: joint angles, limits
- Outputs: within_limits (bool), violations, margin

**`statistical_comparison()`** - Hypothesis testing
- Inputs: two data arrays, metric name
- Outputs: t-test, U-test, Cohen's d results

---

## 6. Experimental Setup

### 6.1 Test Configuration

| Parameter | Value |
|-----------|-------|
| Number of targets | 100,000 |
| Batch size | 5,000 |
| Position error threshold | 1 mm |
| Rotation error threshold | 0.01 rad |
| Numerical solver tolerance | 10⁻⁶ |
| Max iterations (numerical) | 300 |
| Number of restarts | 3 |

### 6.2 System Specifications

Recorded in each benchmark:
- CPU model and architecture
- Physical and logical core count
- Maximum CPU frequency
- Available memory
- Power state (battery/plugged)

### 6.3 Controlled Variables

- Same target set for both solvers
- Same DH parameters
- Same joint limit definitions
- Same success criteria
- Sequential (non-parallel) execution

---

## 7. Output Metrics Summary

| Metric | Unit | Analytical | Numerical |
|--------|------|------------|-----------|
| Position Error | meters | ~10⁻¹⁵ | ~10⁻⁷ |
| Rotation Error | radians | ~10⁻¹⁵ | ~10⁻⁷ |
| Computation Time | ms | ~0.3 | ~8 |
| Success Rate | % | 100 | 100 |
| Joint Limit Compliance | % | ~13 | ~27 |
| Manipulability | - | ~6.0 | ~6.0 |
| Throughput | solves/sec | ~2500 | ~120 |
| Convergence Rate | % | N/A | ~99.96 |

---

## 8. Limitations and Considerations

1. **Analytical Solver:**
   - Solution validity depends on workspace reachability
   - May produce solutions outside joint limits
   - Specific to this robot geometry (not general purpose)

2. **Numerical Solver:**
   - May converge to local minima
   - Performance depends on initial guess quality
   - Iteration count varies with target difficulty

3. **Benchmark:**
   - Wall-clock timing affected by system load
   - Memory measurements are batch-level, not per-solve
   - FLOPS estimates are approximate

4. **Statistical:**
   - P-values may underflow for very large samples
   - Effect sizes more meaningful than p-values for large N
