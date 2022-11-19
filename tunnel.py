import pygame as pg
import numpy as np
import taichi as ti
import taichi_glsl as ts
from taichi_glsl import vec2, vec3

ti.init(arch=ti.cuda)

RES = WIDTH, HEIGHT = vec2(1280, 720)

texture = pg.image.load('img/lava.jpg')
texture_size = texture.get_size()[0]
texture_array = pg.surfarray.array3d(texture).astype(np.float32) / 255


@ti.data_oriented
class PyShader:
	def __init__(self, app):
		self.app = app
		self.screen_array = np.full((WIDTH, HEIGHT, 3), [0,0,0], np.uint8)
		self.screen_field = ti.Vector.field(3, ti.uint8, (WIDTH, HEIGHT))
		self.texture_field = ti.Vector.field(3, ti.float32, texture.get_size())
		self.texture_field.from_numpy(texture_array)

	@ti.kernel
	def render(self, time: ti.float32):
		for frag_coord in ti.grouped(self.screen_field):
			uv = (frag_coord - 0.5 * RES) / RES.y
			col = vec3(0.0)

			uv += vec2(0.2 * ts.sin(time / 2), 0.3 * ts.cos(time / 3))

			phi = ts.atan(uv.y, uv.x)
			rho = ts.length(uv)

			st = vec2(phi / ts.pi, 0.25 / rho)
			st.x += time / 14
			st.y += time / 2
			col += self.texture_field[int(st * texture_size)]

			col *= rho + 0.2
			col += 0.1 /rho * vec3(0.3, 0.1, 0.0)

			col = ts.clamp(col, 0.0, 1.0)
			self.screen_field[frag_coord.x, RES.y - frag_coord.y] = col * 255

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

			[exit() for i in pg.event.get() if i.type == pg.QUIT]
			self.clock.tick(60)
			pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')


if __name__ == '__main__':
	app = App()
	app.run()
