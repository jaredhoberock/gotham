#!/usr/bin/env python

# instantiate the 'gotham context'
import api
g = api.Gotham2()

# set up a unit square we can use for the walls of the box
# four vertices
# then two triangles
# this square lies in the xz plane and points 'up'
unitSquare = ([-0.5, 0,  0.5,
                0.5, 0,  0.5,
                0.5, 0, -0.5,
               -0.5, 0, -0.5],
              [   0, 1,  3,
                  1, 2,  3])

# let's start describing geometry

# back wall

# first 'bind a shader'
# the name is matte, and it has one parameter, 'Kd', the diffuse color.
# The value of the parameter, in this case, grey, should immediately follow the parameter name.
g.material('matte', 'Kd', (0.8, 0.8, 0.8))

# push a matrix onto the stack -- this works just like OpenGL
g.pushMatrix()

# translate the square 'back'
g.translate(0, 0, -1)

# rotate the square so it faces the 'front'
g.rotate(90, 1, 0, 0)

# scale the square so it is two units across
g.scale(2.0,2.0,2.0)

# now send the square's geometry to gotham using the mesh() command
g.mesh(unitSquare[0], unitSquare[1])

# now pop the matrix off the stack
g.popMatrix()


# the other walls are instantiated similarly

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


# create a light source
# to do this, we will use a shader which emits light and make a square-shaped mesh

# light
# use the 'light' shader: this one needs to know how bright it is, and what color it is as well
# the parameter controlling 'brightness' is named power
g.material('light', 'power', (20,20,20))
g.pushMatrix()
g.translate(0, 0.95, 0)
g.scale(0.5, 0.5, 0.5)
g.rotate(180.0, 1.0, 0, 0)
g.mesh(unitSquare[0], unitSquare[1])
g.popMatrix()

# make some shiny balls

# mirror ball
# the mirror shader needs to know how reflective it is
g.material('mirror', 'Kr', (1.0, 1.0, 1.0))

# make a ball using the 'sphere' command
# this command takes 4 arguments:
# the x, y, and z coordinates of the sphere's center and the length of its radius
g.sphere(-0.4, -0.66, -0.15, 0.33)

# glass ball
# the glass shader needs to know three things:
# how reflective it is (as a color)
# how transmissive it is (as a color)
# and an index of refraction (actual glass's index is around 1.5)
g.material('perfectglass', 'Kr', (1.0, 1.0, 1.0), 'Kt', (1.0, 1.0, 1.0), 'eta', 1.5)
g.sphere(0.4,-0.66,0.25,0.33)

# now we need to tell gotham how it should perform the render
# we do this by setting gothams's state with the attribute() command

# first we need to specify how light paths should be created
# we do this by setting the value of a parameter called 'path::sampler'
# there are several of these, and many have parameters, but a straightforward one to use is called 'kajiya'
g.attribute("path::sampler", "kajiya")

# we can specify how long (in vertices) light paths are allowed to grow.
# for the cornell box, 7 is a good compromise between correctness and speed
# notice that even though 7 is a digit, we specify the attribute as a string, for annoying reasons
g.attribute("path::maxlength", "7")

# now we specify which rendering algorithm to use
# let's use traditional 'montecarlo'
# this one takes no additional parameters
g.attribute("renderer::algorithm", "montecarlo")

# let's render to a 512x512 image
(w,h) = (512,512)

# finally we need to set up the camera
# first push a matrix
g.pushMatrix()

# position the center of the camera's aperture (this of this as the eye) at (0,0,3.0)
# look at the point (0,0,-1) in the middle of the back wall of the cornell box
# consider the direction (0,1,0) as 'up'
# the semantics of this call is identical to OpenGL's glLookAt()
g.lookAt( (0,0,3.0), (0,0,-1), (0,1,0) )

# now instantiate a camera
# this call takes three arguments
# the aspect ratio of the image
# the vertical field of view in degrees
# and the distance to the near plane
g.camera(float(w) / h, 60.0, 0.01)

# clean up the matrix stack
g.popMatrix()

# finally, tell gotham to render
# render() takes two arguments
# the dimensions of the image as given in a single tuple
# and the square root of the number of samples to take per pixel
# we want 16 samples per pixel, so we say 4
g.render((w,h), 4)

