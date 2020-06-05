function monte_carlo_simulate(dynamics_func, controller_nnet_address, last_layer_activation, init_x, n_sim, n_timesteps, dt)
    n_states = length(init_x.center)
    min_x = [[Inf64  for n = 1:n_states] for m = 1:n_timesteps]
    max_x = [[-Inf64 for n = 1:n_states] for m = 1:n_timesteps]
    controller = read_nnet(controller_nnet_address, last_layer_activation=last_layer_activation)
    xvec = zeros(n_sim, n_timesteps, n_states)
    for i = 1:n_sim
        x  = rand(n_states)
        x .*= init_x.radius * 2
        x .+= init_x.center - init_x.radius
        for j = 1:n_timesteps
            u = compute_output(controller, x)
            dx = dynamics_func(x, u)
            x = x + dx*dt
            min_x[j] = min.(x, min_x[j])
            max_x[j] = max.(x, max_x[j])
            xvec[i, j, :] = x
        end
    end

    output_sets = [init_x]
    for (m1, m2) in zip(min_x, max_x)
        println(m1, m2)
        push!(output_sets, Hyperrectangle(low=m1, high=m2))
    end
    return output_sets, xvec
end


rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])


function plot_output_sets(output_sets; idx=[1,2], fig=nothing, linewidth=3,
    linecolor=:black, linestyle=:solid, fillalpha=0, fill=:red)

    p = isnothing(fig) ? plot() : fig
    for s in output_sets
        w = s.radius[idx[1]] * 2
        h = s.radius[idx[2]] * 2
        x = s.center[idx[1]] - s.radius[idx[1]]
        y = s.center[idx[2]] - s.radius[idx[2]]
        #plot!(rectangle(w,h,x,y), fillalpha=0.0, kwargs)
        plot!(rectangle(w,h,x,y), fillalpha=fillalpha, fill=fill, legend=nothing,
                   linewidth=linewidth, linecolor=linecolor, linestyle=linestyle)
        xlabel!("x_$(idx[1])")
        ylabel!("x_$(idx[2])")
    end
    return p
end

function plot_output_hist(data, ntime; fig=nothing, idx=[1,2], nbins=:notspecified)
    p = isnothing(fig) ? plot() : fig
    x = data[:, ntime, idx[1]]
    y = data[:, ntime, idx[2]]
    if nbins == :notspecified
        fig = histogram2d!(x, y, legend=false)
    else
        fig = histogram2d!(x, y, nbins=nbins, legend=false)
    end
    return fig
end
