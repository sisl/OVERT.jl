import numpy as np
from funs import single_pendulum, Ex_2, car
from gym_new.pendulum_new import Pendulum1Env, Pendulum2Env

def simulate(sim_func, prop, controller_model, init_set, state_vars, n_simulation, ncheck_invariant, dt, use_env=False):
    for i in range(n_simulation):
        x = []
        n_state = len(init_set.keys())
        for j in range(n_state):
            xj = np.random.uniform(init_set[state_vars[j]][0], init_set[state_vars[j]][1])
            x.append(xj)
        if use_env:
            env = sim_func(x_0=x, dt=dt)
            env.reset()
            for nt in range(ncheck_invariant - 1):
                T = controller_model.predict(env.x.reshape(1, -1)).reshape(-1)
                env.step(T)
                if property_violated(prop, state_vars, env.x):
                    print("at time %d." %nt)
                    return
        else:
            for _ in range(ncheck_invariant - 1):
                T = controller_model.predict(np.array(x).reshape(1, -1)).reshape(-1)
                x = sim_func(x, T, dt)
                if property_violated(prop, state_vars, x):
                    return

    # no violation if we get to here.
    print("No violation was found in %d simulations" % n_simulation)

def property_violated(prop, state_vec, state_vals):
    assert len(state_vec) == len(state_vals)
    for c in prop.constraints:
        assert len(c.monomials) == 1
        type = c.type.type2str[c.type._type]
        idx = state_vec.index(c.monomials[0].var)
        x = state_vals[idx]
        if type == ">":
            if x * c.monomials[0].coeff <= c.scalar :
                print("property violated: x_%d = %0.3f was reached" %(idx, x), end="")
                return True
        elif type == "<":
            if x * c.monomials[0].coeff >= c.scalar :
                print("property violated: x_%d = %0.3f was reached" %(idx, x), end="")
                return True
        else:
            raise(IOError("Property %s is not supported" %type))

    return False



# simulate_property(env_name, n_simulation)
# violation_found = False
#
# for i in range(n_simulation):
#     th = np.random.uniform(init_set[states[0]][0], init_set[states[0]][1])
#     dth = np.random.uniform(init_set[states[1]][0], init_set[states[1]][1])
#     for j in range(ncheck_invariant-1):
#         T = model.predict(np.array([th, dth]).reshape(1,2))[0][0]
#         th, dth = single_pendulum(th, dth, T, dt)
#         if th < p1.scalar or th > p2.scalar:
#             print("violation was found: %0.3f" %th)
#             violation_found = True
#             break
#     if violation_found:
#         break
#
# if not violation_found:
#     print("No violation was found in %d simulations" %n_simulation)




# n_repeat = 10000
# no_vi = True
# for _ in range(n_repeat):
#     x_0_0 = np.random.uniform(init_set[states[0]][0], init_set[states[0]][1])
#     x_0_1 = np.random.uniform(init_set[states[1]][0], init_set[states[1]][1])
#     x_0 = [x_0_0, x_0_1]
#     env = Pendulum1Env(x_0=x_0, dt=dt)
#     env.reset()
#     for time in range(ncheck_invariant-1):
#         # print("time: %d, th=%0.3f, thdot=%0.3f" %(time, env.x[0], env.x[1]))
#         torque = model.predict(env.x.reshape(-1,2)).reshape(1)
#         env.step(torque)
#
#     if property_violated(env, prop):
#         print("***property was violated***")
#         no_vi = False
#         break
#
# if no_vi:
#     print("no violation found in %d simulations" %n_repeat)


# env.render()
