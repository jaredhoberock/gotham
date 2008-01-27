#!/usr/bin/env python
import api

g = api.Gotham2()

g.attribute("path:sampler", "kajiya")
g.attribute("path:maxlength", "10")

g.material('matte', 'Kd', (1,1,1))
points = [-1.0, -1.0, 0,
           1.0, -1.0, 0,
           1.0,  1.0, 0,
          -1.0,  1.0, 0]
triangles = [0, 1, 2,
             0, 2, 3]
g.mesh(points, triangles)

# put a dim light behind the camera
g.material('light', 'power', (20,20,20))
points = [ 1.0, -1.0, 3,
          -1.0, -1.0, 3,
          -1.0,  1.0, 3,
           1.0,  1.0, 3]
g.mesh(points, triangles)

# put a very bright light behind the quad
g.material('light', 'power', (2000000,2000000,2000000))
points = [-1.0, -1.0, -3,
           1.0, -1.0, -3,
           1.0,  1.0, -3,
          -1.0,  1.0, -3]
g.mesh(points, triangles)

(w,h) = (512,512)

g.pushMatrix()
# why is 1.75 the magic number?
g.translate(0,0,1.75)
aspect = float(w) / h
g.camera(aspect, 60.0, 0.0)
g.popMatrix()

g.render((w,h))

