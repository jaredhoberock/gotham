#!/usr/bin/env python
import api
import objtogoth

g = api.Gotham2()

g.attribute("path:sampler", "kajiya")
g.attribute("path:maxlength", "10")

g.material('zoneplate', 'f', 512.0)
points = [-1.0, -1.0, 0,
           1.0, -1.0, 0,
           1.0,  1.0, 0,
          -1.0,  1.0, 0]
triangles = [0, 1, 2,
             0, 2, 3]
g.mesh(points, triangles)

(w,h) = (512,512)

g.pushMatrix()
# why is 1.75 the magic number?
g.translate(0,0,1.75)
aspect = float(w) / h
g.camera(aspect, 45.0, 0.0)
g.popMatrix()

g.render((w,h))

