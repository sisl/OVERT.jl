# simpler high D bound
using Plots
pyplot()
include("/Users/Chelsea/Dropbox/AAHAA/src/OverApprox/OverApprox/src/OverApprox.jl")
using Calculus

function get_grid(x, y)
    X =repeat(reshape(x,1,:),length(y),1)
    Y = repeat(reverse(y), 1, length(x))
    return X,Y
end

function flatten(x)
     return vcat(x...)
end

function sortbycol(x, c)
    return x[sortperm(x[:,c]),:]
end


f2 = [x -> cos(x^2),
      x -> -2*x*sin(x^2),
      x -> -2*(sin(x^2) + 2*(x^2)*cos(x^2)) ]

f4 = [y -> y^2,
      y -> 2*y,
      y -> 2]

funs = [f2, f4]
overapprox_pts = []
for i = 1:length(funs)
  tup = funs[i]
  f = tup[1]
  df = tup[2]
  d2f = tup[3]
  if (i == 2)
      convex = true
      N = 18
  else
      convex = nothing
      N = 2
  end
  dmin = -π
  dmax = π
  LB, UB = Main.OverApprox.overapprox(f, dmin, dmax, N; convex=convex, df = df, d2f = d2f, plot=false)
  LBpts = hcat(flatten(LB[1]), flatten(LB[2]))
  LBpts = sortbycol(LBpts, 1)
  UBpts = hcat(flatten(UB[1]), flatten(UB[2]))
  UBpts = sortbycol(UBpts, 1)
  push!(overapprox_pts, (LBpts, UBpts))
end

# cos(x^2)
xr = range(-π, stop=π, length=1000)
plot(xr, cos.(xr.^2), linewidth=3, label="cos(x^2)", xlims=[-1, 1], ylims=[.9, 1.05])
plot!(overapprox_pts[1][2][:,1], overapprox_pts[1][2][:,2], linewidth=2,label="UB", xlabel="x", title="Overapproximation")
plot!(overapprox_pts[1][1][:,1], overapprox_pts[1][1][:,2], linewidth=2,label="LB")

@assert(all(overapprox_pts[1][2][:,2] .>= cos.(overapprox_pts[1][2][:,1].^2)))

# y^2
plot(overapprox_pts[2][2][:,1], overapprox_pts[2][2][:,2])
yr = range(-π, stop=π, length=200)
plot!(yr,yr.^2)

@assert(all(overapprox_pts[2][2][:,2] .>= overapprox_pts[2][2][:,1].^2))

# UB
x = overapprox_pts[1][2][:,1]
y = overapprox_pts[2][2][:,1]
X,Y = get_grid(x, y)

# approx points
fx = overapprox_pts[1][2][:,2]
fy = overapprox_pts[2][2][:,2]
fX, fY = get_grid(fx, fy)
F = fX + fY

# plot
# actual function
f(x,y) = cos.(x.^2) + y.^2
xr = yr = range(-π, stop=π, length=200)
plot(xr, yr, f, st = :surface, camera=(15,40), xlabel="x", ylabel="y", plot_title="cos(x^2) + y^2")

# UB
plot!(X,Y,F, st = :surface, camera=(15,40), xlabel="x", ylabel="y", plot_title="cos(x^2) + y^2")# alpha= 0.7, color = :blues)

#f(x, y) = begin
#         (3x + y ^ 2) * abs(sin(x) + cos(y))
#     end
# X = repeat(reshape(x, 1, :), length(y), 1)
# Y = repeat(y, 1, length(x))
# Z = map(f, X, Y)
# p1 = contour(x, y, f, fill=true)
#plot(x, y, f, st = :surface, camera=(15,40))


# x = [1,2,3]
# X =repeat(reshape(x,1,:),3,1)
# y = [4,5,6]
# Y = repeat(y, 1, 3)
# F = X.^2 + Y.^2
# plot(X,Y,F, st = :surface, camera=(15,40))

# check validity of bound
for i = 1:length(x)
           for j = 1:length(y)
               xi = X[length(y)-j+1,i]
               yj = Y[length(y)-j+1,i]
               if !(F[length(y)-j+1,i] >= f(xi,yj))
                   println("oh noooo...")
                   println(xi,yj)
                   println("UBf: ",F[length(y)-j+1,i]," f",f(xi,yj))
               end
       end
end
