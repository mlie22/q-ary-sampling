# Sampling q-ary lattices

This repository contains samplers some q-ary lattices. We also include code for the samplers in Espitau et. al.'s work following the pseudocode in their paper. 

## Small samplers

Rather than provide the samplers in their most general form, we provide samplers for the root lattices which are used for comparison in the paper. We compare the time it takes to obtain 100,000 samples. 

## Further documentation

We mainly rely on the discrete Gaussian sampler for the integers provided by Sage. Further documentation can be found here: https://doc.sagemath.org/html/en/reference/stats/sage/stats/distributions/discrete_gaussian_integer.html.