#!

# run script for repeatability for ARCH-COMP NNCS category

# single pendulum experiments
# satisfiability
# for 10 steps
julia1.4 --project="." benchmark/singlepend_satisfiability_small_controller_first10.jl 
julia1.4 --project="." benchmark/singlepend_satisfiability_plot.jl 
# for 40 steps
julia1.4 --project="." benchmark/singlepend_satisfiability_small_controller.jl &
# reachability
julia1.4 --project="." benchmark/singlepend_reachability_small_controller.jl 
julia1.4 --project="." benchmark/singlepend_reachability_plot.jl

# tora example
# reachability
julia1.4 --project="." benchmark/benchmark_tora.jl


# Sherlock benchmark 10 - Car
# reachability
# julia1.4 ...

# ACC
# reachability
# julia1.4 ...
