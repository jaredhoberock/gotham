#!/usr/bin/env python
import api

g = api.Gotham2()

unitSquare = ([-0.5, 0,  0.5,
                0.5, 0,  0.5,
                0.5, 0, -0.5,
               -0.5, 0, -0.5],
              [   0, 1,  3,
                  1, 2,  3])

# back wall
g.material('matte', 'Kd', (0.8, 0.8, 0.8))
g.pushMatrix()
g.translate(0, 0, -1)
g.rotate(90, 1, 0, 0)
g.scale(2.0,2.0,2.0)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

## floor
#g.material('matte', 'Kd', (0.8, 0.8, 0.8))
#g.pushMatrix()
#g.translate(0,-1,0)
#g.scale(2.0, 2.0, 2.0)
#g.mesh(unitSquare[0], unitSquare[1])
#g.popMatrix()

# ceiling
g.material('matte', 'Kd', (0.8, 0.8, 0.8))
g.pushMatrix()
g.translate(0,1,0)
g.scale(2.0, 2.0, 2.0)
g.rotate(180.0, 1.0, 0, 0)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

# left wall
g.material('matte', 'Kd', (0.8, 0.2, 0.2))
g.pushMatrix()
g.translate(-1,0,0)
g.scale(2.0, 2.0, 2.0)
g.rotate(-90, 0, 0, 1)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

# right wall
g.material('matte', 'Kd', (0.2, 0.8, 0.2))
g.pushMatrix()
g.translate(1,0,0)
g.scale(2.0, 2.0, 2.0)
g.rotate(90, 0, 0, 1)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

g.material('phongreflection', 'Kr', (1.0, 1.0, 1.0), 'eta', 2.485, 'exponent', 100000)
g.material('mirror', 'Kr', (1.0, 1.0, 1.0))
g.sphere(-0.4, -0.66, -0.15, 0.33)

# floor
g.material('phongreflection', 'Kr', (0.8, 0.8, 0.8), 'eta', 2.485, 'exponent', 25)
#g.material('matte', 'Kd', (0.8, 0.8, 0.8))
g.pushMatrix()
g.translate(0,-1,0)
g.scale(2.0, 2.0, 2.0)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

g.material('perfectglass', 'Kr', (1.0, 1.0, 1.0), 'Kt', (1.0, 1.0, 1.0), 'eta', 1.5)
#g.material('thinglass', 'Kr', (1.0, 1.0, 1.0), 'Kt', (1.0, 1.0, 1.0))
g.sphere(0.4,-0.66,0.25,0.33)

# lights
g.material('light', 'power', (10,10,10))

g.pushMatrix()
g.translate(-0.5, 0.95, 0)
g.scale(0.5, 0.5, 0.5)
g.rotate(180.0, 1.0, 0, 0)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

g.pushMatrix()
g.translate(0.5, 0.95, 0)
g.scale(0.5, 0.5, 0.5)
g.rotate(180.0, 1.0, 0, 0)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

g.attribute("path::sampler", "kelemen")
#g.attribute("path::sampler", "whitted")
#g.attribute("path::maxlength", "20")
g.attribute("path::maxlength", "10")
#g.attribute("path::russianroulette::function", "always")
g.attribute("renderer::algorithm", "metropolis")
g.attribute("importance::function", "constant")

(w,h) = (512,512)
g.pushMatrix()
g.lookAt( (0,0,3.0), (0,0,-1), (0,1,0) )
g.camera(float(w) / h, 60.0, 0.01)
g.popMatrix()

g.attribute("record::width", str(w))
g.attribute("record::height", str(h))
g.attribute("renderer::spp", "4")
g.render()

