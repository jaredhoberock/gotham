#!/usr/bin/env python
import api
import objtogoth

g = api.Gotham2()

g.material('matte', "Kd", (1.0,1.0,1.0))
(vertices, uv, indices) = objtogoth.objtogoth("../../../data/geometry/obj/rooms.obj")
g.mesh(vertices, uv, indices)

# light
g.material('light', 'power', (3500 * 500,3500 * 500,3500 * 500))
g.sphere(242, 33, -88, 5)

#g.attribute("path::sampler", "kajiya")
#g.attribute("path::sampler", "kelemen")
g.attribute("path::sampler", "simpleforwardrussianroulette")
g.attribute("path::russianroulette::function", "constant")
g.attribute("path::russianroulette::continueprobability", "0.5")
#g.attribute("renderer::targetrays", "60000000")

g.attribute("mutator::strategy", "stratified")
g.attribute("path::maxlength", "10")
g.attribute("renderer::algorithm", "metropolis")
g.attribute("mutator::largestep", "0.001")
#g.attribute("importance::function", "luminanceovervisits")

#g.attribute("importance::function", "constant")
#g.attribute("record::outfile", "rooms_with_roof.kajiya.constant.exr")

#g.attribute("importance::function", "equalvisit")
#g.attribute("renderer::outfile", "rooms_with_roof.kajiya.equalvisit.exr")

#g.attribute("importance::function", "luminanceovervisits")
#g.attribute("record::outfile", "rooms_with_roof.kajiya.luminanceovervisits.exr")

g.attribute("importance::function", "throughputluminance")
g.attribute("record::outfile", "rooms_with_roof.kajiya.throughputluminance.exr")

#g.attribute("importance::function", "luminance")
#g.attribute("record::outfile", "rooms_with_roof.kajiya.luminance.exr")

(w,h) = (800,400)
g.pushMatrix()

g.lookAt( (  29.7108,  18.1446,  -107.85),
          (  30.6369,  18.2885,  -108.199),
          (-0.138807, 0.989526, 0.039636) )

g.camera(float(w) / h, 60.0, 0.01)
g.popMatrix()

g.render((w,h), 3)

