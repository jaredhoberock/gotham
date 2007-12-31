#!/usr/env/bin python

# TODO
# 1. fix bug that if a script shares the same name as a material dll to load,
#    the material may not be loaded correctly
# 2. fix multiple definitions of c++ -> python conversion for Spectrum

import sys
import os
import math
from gotham import *

def normalize(x):
  length = math.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
  return (x[0] / length, x[1] / length, x[2] / length)

def cross(a,b):
  v = (a[1]*b[2] - a[2]*b[1],
       a[2]*b[0] - a[0]*b[2],
       a[0]*b[1] - a[1]*b[0])
  return v

def mul(A, x):
  b0 = A[ 0] * x[0] + A[ 1] * x[1] + A[ 2] * x[2] + A[ 3] * x[3]
  b1 = A[ 4] * x[0] + A[ 5] * x[1] + A[ 6] * x[2] + A[ 7] * x[3]
  b2 = A[ 8] * x[0] + A[ 9] * x[1] + A[10] * x[2] + A[11] * x[3]
  b3 = A[12] * x[0] + A[13] * x[1] + A[14] * x[2] + A[15] * x[3]
  return (b0, b1, b2, b3)

class Gotham2(Gotham):
  # standard shaderpaths
  shaderpaths = ['.']
  try:
    shaderpaths += [os.environ['GOTHAMHOME'] + '/shaders']
  except:
    print 'Warning: $GOTHAMHOME undefined! Some shaders may not be found.'

  def material(self, name, *parms):
    # XXX this is getting ugly
    # add shaderpaths to os.path temporarily
    oldpath = sys.path
    sys.path += self.shaderpaths
    try:
      # import the material
      module = __import__(name)
      # create a new material
      m = module.createMaterial()
      # set parameters
      for i in range(0, len(parms), 2):
        p = parms[i]
        val = parms[i+1]
        try:
          setMethod = getattr(m, 'set_' + p)
          try:
            # first try to set it as if it were a 3-vector
            setMethod(val[0], val[1], val[2])
          except:
            try:
              # try a scalar instead
              setMethod(val)
            except:
              print 'Warning: %s has unknown type; material parameter left undefined.'
        except:
          print 'Warning: "%s" is not a parameter of material "%s"!' % (p, name)
      del module
      Gotham.material(self, m)
      result = True
    except:
      print "Unable to find material '%s'." % name
      result = False
    sys.path = oldpath
    return result

  def mesh(self, *args):
    # validate arguments
    if (len(args) != 2 and len(args) != 3):
      raise ValueError, "mesh() expects either two (points,indices) or three (points,parms,indices) arguments."
    # convert to vectors
    points = args[0]
    pointsvec = vector_float()
    pointsvec[:] = points
    if len(args) == 2:
      faces = args[1]
    else:
      faces = args[2]
    # validate faces
    if (len(faces) % 3) != 0:
      raise ValueError, "Triangle list not a multiple of 3!"
    i = 0
    for v in faces:
      if v >= len(points) / 3:
        raise ValueError, "Triangle %d refers to non-vertex!" % (i/3)
      i += 1
    facesvec = vector_uint()
    facesvec[:] = faces
    if len(args) == 3:
      parmsvec = vector_float()
      parmsvec[:] = args[1]
      Gotham.mesh(self, pointsvec, parmsvec, facesvec)
    else:
      Gotham.mesh(self, pointsvec, facesvec)

  def multMatrix(self, m):
    matrix = vector_float()
    matrix[:] = m
    Gotham.multMatrix(self, matrix)

  def loadMatrix(self, m):
    matrix = vector_float()
    matrix[:] = m
    Gotham.loadMatrix(self, matrix)

  def render(self, *args):
    if len(args) > 0:
      print "render((w,h), spp): Warning: using arguments with this function is deprecated."
      print "Please use render() instead."
    if len(args) == 1:
      print "Error: too few parameters to render()."
      return
    if len(args) > 2:
      print "Error: too many parameters to render()."
      return
    elif len(args) == 2:
      Gotham.attribute(self, "record::width", str(args[0][0]))
      Gotham.attribute(self, "record::height", str(args[0][1]))
      Gotham.attribute(self, "renderer::spp", str(args[1]))
    Gotham.render(self)

  def lookAt(self, eye, center, up):
    # see gluLookAt man page: we construct the
    # transpose of their matrix and inverse of their
    # translation
    f = (float(center[0] - eye[0]),
         float(center[1] - eye[1]),
         float(center[2] - eye[2]))
    f = normalize(f)
    up = normalize(up)
    s = cross(f,up)
    u = cross(s,f)
    M = vector_float()
    M[:] = [s[0],  u[0], -f[0], 0.0,
            s[1],  u[1], -f[1], 0.0,
            s[2],  u[2], -f[2], 0.0,
             0.0,   0.0,   0.0, 1.0] 
    Gotham.translate(self, eye[0], eye[1], eye[2])
    Gotham.multMatrix(self, M)

  def pointlight(self, position, power, radius = 0.0005):
    Gotham.pushAttributes(self)
    Gotham2.material(self, 'light', 'power', power)
    Gotham.sphere(self, position[0], position[1], position[2], radius)
    Gotham.popAttributes(self)

  def camera(self, aspect, fovy, apertureRadius):
    # create a small rectangle for the aperture
    # centered at the 'eye' point
    epsilon = 0.0005
    apertureRadius += epsilon
    # the aperture starts out as a unit square with normal pointing in the 
    # -z direction
    points = [-0.5, -0.5, 0,  -0.5, 0.5, 0,  0.5, 0.5, 0,  0.5, -0.5, 0]
    triangles = [0, 1, 3,  1, 2, 3]
    Gotham.pushMatrix(self)
    Gotham.scale(self, apertureRadius/2, apertureRadius/2, apertureRadius/2)
    # compute the center of the aperture in world coordinates by multiplying
    # (0,0,0) by the current matrix
    # assign the perspective material
    m = vector_float()
    Gotham.getMatrix(self, m)
    c = mul(m, (0,0,0,1))
    up = mul(m, (0,1,0,0))
    up = (up[0],up[1],up[2])
    up = normalize(up)
    right = mul(m, (1,0,0,0))
    right = (right[0],right[1],right[2])
    right = normalize(right)
    look = mul(m, (0,0,-1,0))
    look = (look[0],look[1],look[2])
    look = normalize(look)
    Gotham.pushAttributes(self)
    # convert to radians
    fovyRadians = fovy * (math.pi/180.0)
    Gotham2.material(self,
                     'perspective',
                     'aspect',aspect,
                     'fovy',fovyRadians,
                     'center',(c[0],c[1],c[2]),
                     'up', up,
                     'right', right,
                     'look', look)
    # name the camera
    Gotham.attribute(self, "name", "camera")
    Gotham2.mesh(self, points, triangles)
    Gotham.popAttributes(self)
    Gotham.popMatrix(self)
    # hint to the viewer after we've popped the attributes
    # XXX we really need a way to pass types besides strings
    Gotham.attribute(self, "viewer::fovy",  str(fovy))
    Gotham.attribute(self, "viewer::eyex",  str(c[0]))
    Gotham.attribute(self, "viewer::eyey",  str(c[1]))
    Gotham.attribute(self, "viewer::eyez",  str(c[2]))
    Gotham.attribute(self, "viewer::upx",   str(up[0]))
    Gotham.attribute(self, "viewer::upy",   str(up[1]))
    Gotham.attribute(self, "viewer::upz",   str(up[2]))
    Gotham.attribute(self, "viewer::lookx", str(look[0]))
    Gotham.attribute(self, "viewer::looky", str(look[1]))
    Gotham.attribute(self, "viewer::lookz", str(look[2]))


