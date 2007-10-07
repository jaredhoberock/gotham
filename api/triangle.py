#!/usr/bin/env python
import api
import objtogoth

g = api.Gotham2()

g.pushMatrix()
g.rotate(180, 0, 0, 1)
g.material('matte', 'Kd', (0,0,1))
(points, triangles) = objtogoth.objtogoth('/home/jared/dev/data/geometry/obj/tri.obj')
g.mesh(points, triangles)
g.popMatrix()

(w,h) = (512,512)

g.pushMatrix()
g.translate(0,0,1)
aspect = float(w) / h
g.camera(aspect, 45.0, 0.0)
g.popMatrix()

g.render((w,h))

