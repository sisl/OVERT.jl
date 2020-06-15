function setpaths()
  rootdir = @__DIR__
  mipdir = joinpath(rootdir,"MIP")
  overtdir = joinpath(rootdir,"OverApprox")
  push!(LOAD_PATH, mipdir, overtdir)
end
setpaths();
