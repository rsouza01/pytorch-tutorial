# Maxwell-Boltzmann Simulation


# 🧪 Simulation-Theory Connection Checklist

## ✅ 1. System Initialization
- [ ] Initialize particles with random positions and velocities.
- [ ] Ensure total momentum and energy are conserved.
- [ ] Visualize initial state (optional but helpful).

## 📊 2. Kinetic Theory Validation
- [ ] Compute average kinetic energy per particle.
- [ ] Calculate temperature using:
  $$T = \frac{2}{d k_B} \left\langle E_k \right\rangle,$$
    (with \( d = 2 \) for 2D)
  .
- [ ] Compare simulated temperature with expected value.

## 📈 3. Velocity Distribution
- [ ] Create a histogram of particle speeds.
- [ ] Overlay the theoretical 2D Maxwell-Boltzmann distribution:
  $$
  f(v) = \frac{m}{k_B T} v \exp\left(-\frac{mv^2}{2k_B T} \right)
  $$
- [ ] Track how the distribution evolves over time.

## 🔄 4. Entropy and Equilibration
- [ ] Bin particle energies or speeds.
- [ ] Compute entropy:
  $$
  S = -k_B \sum_i p_i \ln p_i
  $$
- [ ] Plot entropy vs. time to observe approach to equilibrium.

## 🧭 5. Time Reversibility Test
- [ ] At a chosen time, reverse all particle velocities.
- [ ] Observe whether the system retraces its path.
- [ ] Reflect on implications for microscopic vs. macroscopic behavior.

## 📐 6. Pressure (Optional if you have walls)
- [ ] Measure momentum transfer to walls over time.
- [ ] Estimate pressure and compare with:
  $$
  P = \frac{N k_B T}{V}
  $$

## 🧠 7. Theoretical Reflection
- [ ] Relate simulation results to the Boltzmann H-theorem.
- [ ] Discuss how the simulation illustrates the second law of thermodynamics.
- [ ] Explore the role of distinguishability in entropy and statistics.


## References

- GPU Optimized: https://www.youtube.com/watch?v=iSEAidM-DDI
