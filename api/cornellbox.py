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


# floor
g.material('matte', 'Kd', (0.8, 0.8, 0.8))
g.pushMatrix()
g.translate(0,-1,0)
g.scale(2.0, 2.0, 2.0)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

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

# light
g.material('light', 'power', (20,20,20))
g.pushMatrix()
g.translate(0, 0.95, 0)
g.scale(0.5, 0.5, 0.5)
g.rotate(180.0, 1.0, 0, 0)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

#g.attribute("path::sampler", "kajiya")
#g.attribute("path::sampler", "simplebidirectional")
#g.attribute("path::sampler", "kelemen")
#g.attribute("path::sampler", "simplerussianroulette")
g.attribute("path::sampler", "simpleforwardrussianroulette")
#g.attribute("renderer::algorithm", "metropolis")
#g.attribute("mutator::strategy", "stratified")
#g.attribute("importance::function", "normalized")
g.attribute("renderer::algorithm", "montecarlo")
g.attribute("path::maxlength", "7")
#g.attribute("path::maxlength", "20")
#g.attribute("path::maxlength", "32")
#g.attribute("path::russianroulette::function", "maxoverspectrum")
g.attribute("path::russianroulette::function", "constant")
g.attribute("path::russianroulette::continueprobability", "0.5")

(w,h) = (512,512)
#(w,h) = (256,256)
#(w,h) = (128,128)
g.pushMatrix()
g.lookAt( (0,0,3.0), (0,0,-1), (0,1,0) )
#g.lookAt( (0,0,2.75), (0,0,-1), (0,1,0) )
g.camera(float(w) / h, 60.0, 0.01)
g.popMatrix()

g.render((w,h), 4)
#g.render((w,h), 10)
#g.render((w,h), 32)

