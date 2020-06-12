# simple wrapper for simulation. takes what comes out of gym.env.make('envstr')

class SimpleWrapper():
	def __init__(self, env):
		self.env = env
		self.reset()

	def set_state(self, s):
		self.env.env.state = s

	def get_state(self):
		return self.env.env.state

	def reset(self):
		return self.env.reset()

	def render(self):
		return self.env.env.render()

	def step(self, action):
		return self.env.step(action)


