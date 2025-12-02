# Slide 1: Methodology

## Fast Trajectory Recovery & Smoothing

### 1. The Challenge
*   **Old Approach (Local Optimization)**: Optimized each 10-step segment independently.
*   **Issue**: Lacked continuity constraints, resulting in "jittery" parameters and unnatural steering inputs.

### 2. The Solution: Hybrid Recovery
*   **Step 1: Local Reconstruction**: Fast, parallel optimization to fit raw data points.
*   **Step 2: Gaussian Smoothing**: Applied a low-pass filter ($\sigma=2.0$) to the recovered parameter sequence.
*   **Why it works**: Effectively removes high-frequency noise while preserving the overall trajectory shape.

### 3. Key Advantages
*   **Efficiency**: reduced processing time from **days** to **<10 minutes** (O(N) complexity).
*   **Scalability**: Can handle massive datasets (8GB+) effortlessly.

### 4. Limitations
*   **Dynamics**: Smoothing is applied in parameter space, which may slightly deviate from strict vehicle dynamics.
*   **Details**: Rapid, intentional maneuvers might be over-smoothed.

---

# Slide 2: Results

## Significant Smoothness Improvements

### Key Metrics (Lower is Better)
*   **Mean Yaw Rate**: **97.7%** Improvement (Stable heading)
*   **Curvature Variance**: **74.1%** Improvement (No sharp turns)
*   **Yaw Continuity**: **91.5%** Improvement (Seamless transitions)
*   **Mean Jerk**: **54.0%** Improvement (Smoother acceleration)

### Trade-offs
*   **Smoothness vs Accuracy**: We gain **>90% smoothness** (Yaw/Curvature) by accepting a minimal **~0.2m position error**.
*   **Speed vs Precision**: We achieve **100x speedup** (days $\to$ minutes) by using parameter-space smoothing instead of full dynamic optimization.
*   **Result**: A highly efficient, physically consistent expert dataset suitable for large-scale training.
