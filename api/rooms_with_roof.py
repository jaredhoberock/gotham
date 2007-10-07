#!/usr/bin/env python
import api
import objtogoth

g = api.Gotham2()

g.material('matte', "Kd", (1.0,1.0,1.0))
(vertices, indices) = objtogoth.objtogoth("/home/jared/dev/data/geometry/obj/rooms.obj")
g.mesh(vertices, indices)

# light
g.material('light', 'power', (3500 * 500,3500 * 500,3500 * 500))
g.sphere(242, 33, -88, 5)

#g.attribute("path::sampler", "kelemen")
#g.attribute("path::sampler", "kajiya")
g.attribute("path::sampler", "simpleforwardrussianroulette")
#g.attribute("path::sampler", "simplebidirectionalrussianroulette")
g.attribute("mutator::strategy", "stratified")
g.attribute("path::maxlength", "10")
#g.attribute("path::maxlength", "20")
g.attribute("renderer::algorithm", "metropolis")
g.attribute("mutator::largestep", "0.001")
g.attribute("path::russianroulette::function", "constant")
g.attribute("path::russianroulette::continueprobability", "0.5")
g.attribute("renderer::output", "rooms_with_roof.simpleforwardrussianroulette.32x32.exr")

(w,h) = (800,400)
g.pushMatrix()

g.lookAt( (  29.7108,  18.1446,  -107.85),
          (  30.6369,  18.2885,  -108.199),
          (-0.138807, 0.989526, 0.039636) )

g.camera(float(w) / h, 45.0, 0.01)
g.popMatrix()

#g.render((w,h), 4)
#g.render((w,h), 10)
g.render((w,h), 32)
#g.render((w,h), 64)

