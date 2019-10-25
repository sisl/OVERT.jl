if @isdefined(run_before)
    ENV["PYTHON"] = "/Users/Chelsea/miniconda2/envs/rllab3/bin/python3"
    ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
    using Pkg
    Pkg.build("PyCall")
    run_before = true
end

using PyCall
using JuliaInterpreter
push!(JuliaInterpreter.compiled_modules, PyCall)

pushfirst!(PyVector(pyimport("sys")."path"), "")
pushfirst!(PyVector(pyimport("sys")."path"), "../gym")

# begin actual program script
include("overest_new.jl")
using Debugger

env_mod = pyimport("gym.envs.classic_control.my_pendulum")
env = env_mod.MyPendulumEnv()

# bound pendulum
f = env.nonlinear_part
df = env.d_nonlinear_part
d2f = env.dd_nonlinear_part
a = -π
b = π
N = 6

# compute upper bound
(x_UB,y_UB) = bound(f, a, b, N; lowerbound=false, df=df, d2f=d2f)

(x_LB,y_LB) = bound(f, a, b, N; lowerbound=true, df=df, d2f=d2f)

# transform into ReLUs
