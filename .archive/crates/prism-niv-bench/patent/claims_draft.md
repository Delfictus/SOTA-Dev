## Independent Claims

1. A computer-implemented method for predicting cryptic epitopes comprising:
   a) Receiving three-dimensional coordinates of a viral protein structure;
   b) Computing eigenmode displacement vectors using normal mode analysis;
   c) Processing displacement vectors through a leaky integrate-and-fire
      neural network to generate spiking features;
   d) Discretizing features into a 6-dimensional state space;
   e) Using a Q-learning agent to predict cryptic residue labels.

2. A system for high-throughput cryptic epitope screening comprising:
   a) A GPU executing fused CUDA kernels for batch structure processing;
   b) A dendritic reservoir computing module for temporal feature integration;
   c) A reinforcement learning agent trained on ground truth epitope data.

## Dependent Claims

3. The method of claim 1, wherein the viral protein is a paramyxovirus
   glycoprotein selected from the group consisting of Nipah virus G protein,
   Nipah virus F protein, and Hendra virus F protein.

4. The method of claim 1, wherein the spiking features include spike density,
   inter-spike interval, and spike timing relative to conformational sampling.
