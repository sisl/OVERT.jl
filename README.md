# OVERT.jl

[![Build Status](https://github.com/sisl/OVERT.jl/workflows/CI/badge.svg)](https://github.com/sisl/OVERT.jl/actions)
[![codecov](https://codecov.io/gh/sisl/OVERT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sisl/OVERT.jl)


This repo contains a julia implementation for the [OVERT algorithm](https://arxiv.org/abs/2108.01220). Overt provides a relational piecewise-linear overapproximation of any multi-dimensional function. 
The overapproximation is useful for verifying systems with nonlinear dynamics. 
In particular, we used OVERT for verifying nonlinear dynamical systems that are controlled by neural networks. See the [OVERTVerify package](https://github.com/sisl/OVERTVerify.jl).

The output of OVERT is a list of equality and inequality constraints that may include nonlinear operations like `min` and `max`. Overt is guaranteed to identify the tightest piecewise linear overapproximation. In addition, the OVERT algorithm has a linear complexity in the input dimension of the function.

## Installation
```
] add OVERT
```


## Usage
```julia
using OVERT

func = :(sin(x + y) * exp(z))
range_dict = Dict(:x => [1., 2.], :y => [-1., 1.], :z => [0., 1.])
o1 = overapprox(func, range_dict)

```
The output is
```julia
output = v_24
v_1 == x + y
v_2 == 0.01 * max(0, -1.006 (v_1 - 0.994)) + 1.004 max(0, min(1.006 (v_1 - 0.0), -0.883 (v_1 - 2.126))) + 1.015 max(0, min(0.883 (v_1 - 0.994), -1.144 (v_1 - 3.0))) + 0.151max(0, 1.145 (v_1 - 2.126))
v_3 == -0.01 max(0, -0.923 (v_1 - 1.083)) + 0.873 max(0, min(0.923 (v_1 - 0.0), -1.130 (v_1 - 1.968))) + 0.912 max(0, min(1.130 (v_1 - 1.083), -0.970 (v_1 - 3.0))) + 0.131 max(0, 0.967 (v_1 -1.967))
v_5 == 1.01 max(0, -2.691 (z - 0.371)) + 1.46 max(0, min(2.69 (z - 0.0), -3.02(z - 0.702))) + 2.028 max(0, min(3.024 (z - 0.37), -3.35 (z - 1.0))) + 2.72 max(0, 3.357 (z - 0.702))
v_6 == 0.99 max(0, -3.387 (z - 0.295)) + 1.28 max(0, min(3.387 (z - 0.0), -2.03 (z - 0.788))) + 2.132 max(0, min(2.028
(z - 0.295), -4.72(z - 1.0))) + 2.708 max(0, 4.72 (z - 0.788))
v_8 == (v_4 - -0.01) / 1.025 + 0.1
v_9 == (v_7 - 0.99) / 1.738+ 0.1
v_10 == -2.292 max(0, -11.258 (v_8 - 0.19)) -1.404 max(0, min(11.259 (v_8 - 0.1), -2.138(v_8 - 0.657))) -0.298 max(0, min(2.13 (v_8 - 0.189), -2.255 (v_8 - 1.1))) + 0.105 max(0, 2.255(v_8 - 0.656))
v_11 == -2.31max(0, -5.60 (v_8 - 0.278)) -1.289 max(0, min(5.60 (v_8 - 0.1), -3.129 (v_8 - 0.598))) -0.524 max(0, min(3.129 (v_8 - 0.278), -1.992(v_8 - 1.1))) + 0.0853 max(0, 1.99 (v_8 - 0.598))
v_13 == -2.292 max(0, -11.258 (v_9 - 0.189)) -1.404 max(0, min(11.258 (v_9 - 0.1), -2.138 (v_9 - 0.656))) -0.298 max(0, min(2.138 (v_9 - 0.189), -2.255 (v_9 - 1.1))) + 0.105 max(0, 2.255 (v_9 - 0.657))
v_14 == -2.312 max(0, -5.603 (v_9 - 0.278)) -1.288 max(0, min(5.603 (v_9 - 0.1), -3.129 (v_9 - 0.598))) -0.524 max(0, min(3.129 (v_9 - 0.278), -1.99 (v_9 - 1.1))) + 0.085 max(0, 1.99 (v_9 - 0.598))
v_16 == v_12 + v_15
v_17 == 0.0198 max(0, -0.398 (v_16+2.115)) + 0.130 max(0, min(0.398 (v_16 +4.625), -0.725 (v_16 + 0.736))) + 0.488 max(0, min(0.725(v_16 + 2.115), -1.056 (v_16 - 0.210))) + 1.244 max(0, 1.056(v_16 +0.736))
v_18 == 0.024 max(0, min(0.397 (v_16 +4.625), -0.566 (v_16+0.340))) + 0.544 max(0, min(0.566 (v_16 + 2.106), -1.814 (v_16 - 0.211))) + 1.224 max(0, 1.814(v_16 + 0.340))
v_20 == 1.783v_19
v_21 == -0.191v_9
v_22 == v_20 + v_21
v_23 == 0.913v_8 - 0.098
v_24 == v_22 + v_23
v_3 ≦ v_4
v_4 ≦ v_2
v_6 ≦ v_7
v_7 ≦ v_5
v_11 ≦ v_12
v_12 ≦ v_10
v_14 ≦ v_15
v_15 ≦ v_13
v_18 ≦ v_19
v_19 ≦ v_17
```

Here is an approximation of tanh, displayed graphically:

![tanh-1](https://user-images.githubusercontent.com/14879690/128933192-10c4f4f0-dceb-43f2-b85d-4ac4c69c6bf0.png)

The bounds are very tight despite only using two line segments per region of uniform convexity because they are fitted to minimize the area between the function and the bound.

See plots/Methods Section.ipynb for more examples.

### Citation
```
@misc{sidrane2021overt,
      title={OVERT: An Algorithm for Safety Verification of Neural Network Control Policies for Nonlinear Systems}, 
      author={Chelsea Sidrane and Amir Maleki and Ahmed Irfan and Mykel J. Kochenderfer},
      year={2021},
      eprint={2108.01220},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
