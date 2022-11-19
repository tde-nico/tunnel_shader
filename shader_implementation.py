import pygame as pg
import numpy as np
import taichi as ti
import taichi_glsl as ts
from taichi_glsl import vec2, vec3

ti.init(arch=ti.cuda)

RES = WIDTH, HEIGHT = 1280, 720


@ti.data_oriented
class PyShader:
	def __init__(self, app):
		self.app = app
		self.screen_array = np.full((WIDTH, HEIGHT, 3), [0,0,0], np.uint8)
		self.screen_field = ti.Vector.field(3, ti.uint8, (WIDTH, HEIGHT))

	@ti.kernel
	def render(self, time: ti.float32):
		for frag_coord in ti.grouped(self.screen_field):
			uv = frag_coord / RES
			col = 0.5 +0.5 * ts.cos(time + vec3(uv.x, uv.y, uv.x) + vec3(0.0, 2.0, 4.0))
			self.screen_field[frag_coord.x, RES[1] - frag_coord.y] = col * 255

	def update(self):
		time = pg.time.get_ticks() * 1e-3
		self.render(time)
		self.screen_array = self.screen_field.to_numpy()

	def draw(self):
		pg.surfarray.blit_array(self.app.screen, self.screen_array)

	def run(self):
		self.update()
		self.draw()


class App:
	def __init__(self):
		self.screen = pg.display.set_mode(RES)
		self.clock = pg.time.Clock()
		self.shader = PyShader(self)

	def run(self):
		while True:
			self.shader.run()
			pg.display.flip()

			[exit() for i in pg.event.get() if i == pg.QUIT]
			self.clock.tick(60)
			pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')


if __name__ == '__main__':
	app = App()
	app.run()
