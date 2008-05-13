#!/usr/env/bin python

# TODO
# 1. fix bug that if a script shares the same name as a material dll to load,
#    the material may not be loaded correctly
# 2. fix multiple definitions of c++ -> python conversion for Spectrum

import sys
import os
import math
from libgotham import *
import inspect

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

# define a class named 'PyGotham'
class PyGotham:
  # standard shaderpaths
  shaderpaths = ['.']
  try:
    shaderpaths += [os.path.join(os.environ['GOTHAMHOME'], 'shaders')]
  except:
    print 'Warning: $GOTHAMHOME undefined! Some shaders may not be found.'

  def __init__(self):
    # by default, the subsystem is plain old Gotham
    self.__subsystem = self.__createSubsystem("Gotham")
    self.attribute("renderer:subsystem", "Gotham")

    # include the directory containing this script
    # in Python's search path
    dir = os.path.dirname(inspect.getabsfile(mul))
    sys.path += [dir]

  def __createSubsystem(self, name, copyFrom = None):
    result = None
    # try to import every file in this directory
    # look for the first one with a type of Gotham which matches name
    dir = os.path.dirname(inspect.getabsfile(mul))

    # try importing each file as a module
    for file in os.listdir(dir):
      fileBasename = os.path.splitext(file)[0]
      try:
        module = __import__(fileBasename)
        if copyFrom == None:
          # call the null constructor
          # g = module.name()
          exec "result = module." + name + "()"
        else:
          # call the copy constructor
          # g = module.name(copyFrom)
          exec "result = module." + name + "(copyFrom)"
        del module
      except:
        pass
      # stop at the first thing we were able to create
      if result != None:
        break;
    return result

  def pushMatrix(self):
    return self.__subsystem.pushMatrix()

  def popMatrix(self):
    return self.__subsystem.popMatrix()

  def translate(self, tx, ty, tz):
    return self.__subsystem.translate(tx,ty,tz)

  def rotate(self, degrees, rx, ry, rz):
    return self.__subsystem.rotate(degrees, rx, ry, rz)

  def scale(self, sx, sy, sz):
    return self.__subsystem.scale(sx, sy, sz)

  def getMatrix(self, m):
    return self.__subsystem.getMatrix(m)

  def sphere(self, cx, cy, cz, radius):
    return self.__subsystem.sphere(cx, cy, cz, radius)

  def pushAttributes(self):
    return self.__subsystem.pushAttributes()

  def popAttributes(self):
    return self.__subsystem.popAttributes()

  def attribute(self, name, value):
    if value == "False":
      return self.__subsystem.attribute(name, str("false"))
    elif value == "True":
      return self.__subsystem.attribute(name, str("true"))
    else:
      return self.__subsystem.attribute(name, str(value))

  def getAttribute(self, name):
    return self.__subsystem.getAttribute(name)

  def material(self, name, *parms):
    # XXX this is getting ugly
    # add shaderpaths to os.path temporarily
    oldpath = sys.path
    sys.path += self.shaderpaths

    #try:
    # import the material
    module = __import__(name)
    # create a new material
    m = module.createMaterial()

    parmDict = {}
    if len(parms) > 1:
      for i in range(0, len(parms), 2):
        parmDict[parms[i]] = parms[i+1]
    elif len(parms) == 1:
      parmDict = parms[0]

    # set each parameter
    if parmDict != {}:
      for p, val in parmDict.iteritems():
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
    # send the material to the subsystem
    self.__subsystem.material(m)
    result = True
    #except:
    #  print "Unable to find material '%s'." % name
    #  result = False
    # restore paths
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
    if len(faces) == 0:
      print 'mesh(): Warning: empty mesh detected.'
      return
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
      return self.__subsystem.mesh(pointsvec, parmsvec, facesvec)
    else:
      return self.__subsystem.mesh(pointsvec, facesvec)

  def multMatrix(self, m):
    matrix = vector_float()
    matrix[:] = m
    return self.__subsystem.multMatrix(matrix)

  def loadMatrix(self, m):
    matrix = vector_float()
    matrix[:] = m
    return self.__subsystem.loadMatrix(matrix)

  def render(self, *args):
    if len(args) > 0:
      print "render((w,h), spp): Warning: using arguments with this function is deprecated."
      print "Please use render() instead."
    if len(args) == 1:
      PyGotham.attribute(self, "record:width", str(args[0][0]))
      PyGotham.attribute(self, "record:height", str(args[0][1]))
    if len(args) > 2:
      print "Error: too many parameters to render()."
      return
    elif len(args) == 2:
      PyGotham.attribute(self, "record:width", str(args[0][0]))
      PyGotham.attribute(self, "record:height", str(args[0][1]))
      PyGotham.attribute(self, "renderer:spp", str(args[1]))

    # finalize the subsystem
    subsys = self.getAttribute("renderer:subsystem")
    if subsys != "":
      # do we need to change subsystems?
      if type(self.__subsystem).__name__ != subsys:
        try:
          # create a copy using the new subsystem type
          self.__subsystem = self.__createSubsystem(subsys, self.__subsystem)
        except:
          print 'Warning: Could not create subsystem named', subsys
    return self.__subsystem.render()

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
    self.translate(eye[0], eye[1], eye[2])
    self.multMatrix(M)

  def pointlight(self, position, power, radius = 0.0005):
    self.pushAttributes()
    self.material('light', 'power', power)
    self.sphere(position[0], position[1], position[2], radius)
    self.popAttributes()

  def camera(self, aspect, fovy, apertureRadius):
    # create a small rectangle for the aperture
    # centered at the 'eye' point
    epsilon = 0.0005
    apertureRadius += epsilon
    # the aperture starts out as a unit square with normal pointing in the 
    # -z direction
    points = [-0.5, -0.5, 0,  -0.5, 0.5, 0,  0.5, 0.5, 0,  0.5, -0.5, 0]
    uv = [0,0,  1,0,  1,1,  0,1]
    triangles = [0, 1, 3,  1, 2, 3]
    self.pushMatrix()
    self.scale(apertureRadius/2, apertureRadius/2, apertureRadius/2)
    # compute the center of the aperture in world coordinates by multiplying
    # (0,0,0) by the current matrix
    # assign the perspective material
    m = vector_float()
    self.getMatrix(m)
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
    self.pushAttributes()
    # convert to radians
    fovyRadians = fovy * (math.pi/180.0)
    # compute the location of the lower-left corner of the viewport,
    # in world coordinates
    near = 1.0 / math.tan(0.5 * fovyRadians)
    #ll = c + near * look - aspect * right - up
    ll = (c[0] + near * look[0] - aspect * right[0] - up[0],
          c[1] + near * look[1] - aspect * right[1] - up[1],
          c[2] + near * look[2] - aspect * right[2] - up[2])
    self.material('perspective',
                  'aspect',aspect,
                  'fovy',fovyRadians,
                  'lowerLeft', ll)
    # name the camera
    self.attribute("name", "camera")
    self.mesh(points, uv, triangles)
    self.popAttributes()
    self.popMatrix()
    # hint to the viewer after we've popped the attributes
    # XXX we really need a way to pass types besides strings
    self.attribute("viewer:fovy",  str(fovy))
    self.attribute("viewer:eyex",  str(c[0]))
    self.attribute("viewer:eyey",  str(c[1]))
    self.attribute("viewer:eyez",  str(c[2]))
    self.attribute("viewer:upx",   str(up[0]))
    self.attribute("viewer:upy",   str(up[1]))
    self.attribute("viewer:upz",   str(up[2]))
    self.attribute("viewer:lookx", str(look[0]))
    self.attribute("viewer:looky", str(look[1]))
    self.attribute("viewer:lookz", str(look[2]))

  def parse(self, lines):
    # XXX we can think about passing each line
    #     to a super-efficient parser in C++
    #     rather than calling Python's exec
    #     because it is slow
    numLines = len(lines)
    # add one to avoid modulo by zero
    fivePercent = numLines / 20 + 1
    lineNumber = 0

    print 'Parsing...'
    for line in lines:
      if lineNumber % fivePercent == 0:
        print 'Progress: ' +  str(100 * float(lineNumber)/numLines) + '%\r',
        sys.stdout.flush()
      # first see if we can parse it quickly in c++
      if not Gotham.parseLine(self, line):
        # each line depends on 'g' being defined as some Gotham object
        g = self
        exec line in globals()
      lineNumber += 1
    print '\nDone.'

  def unitcube(self):
    unitSquare = ([-0.5, 0,  0.5,
                    0.5, 0,  0.5,
                    0.5, 0, -0.5,
                   -0.5, 0, -0.5],
                  [   0, 0,
                      1, 0,
                      1, 1,
                      0, 1],
                  [   0, 1,  3,
                      1, 2,  3])

    # front wall
    self.pushMatrix()
    self.translate(0, 0, 0.5)
    self.rotate(90, 1, 0, 0)
    self.mesh(unitSquare[0], unitSquare[1], unitSquare[2])
    self.popMatrix()

    # left wall
    self.pushMatrix()
    self.translate(-0.5,0,0)
    self.rotate(90, 0, 0, 1)
    self.mesh(unitSquare[0], unitSquare[1], unitSquare[2])
    self.popMatrix()

    # right wall
    self.pushMatrix()
    self.translate(0.5,0,0)
    self.rotate(-90, 0, 0, 1)
    self.mesh(unitSquare[0], unitSquare[1], unitSquare[2])
    self.popMatrix()

    # back wall
    self.pushMatrix()
    self.translate(0, 0, -0.5)
    self.rotate(-90, 1, 0, 0)
    self.mesh(unitSquare[0], unitSquare[1], unitSquare[2])
    self.popMatrix()

    # ceiling
    self.pushMatrix()
    self.translate(0, 0.5,0)
    self.mesh(unitSquare[0], unitSquare[1], unitSquare[2])
    self.popMatrix()

    # floor
    self.pushMatrix()
    self.translate(0, -0.5,0)
    self.rotate(180, 1, 0, 0)
    self.mesh(unitSquare[0], unitSquare[1], unitSquare[2])
    self.popMatrix()

def __copyright():
  print 'Gotham 0.1'
  print '(c) Copyright 2007-2008 Jared Hoberock. All Rights Reserved.'

# print copyright info as soon as this is imported
__copyright()

# wrap up the api in a singleton
# create the 'canonical' Gotham instance
# this is not technically a singleton but
# the idea is to make it work like one
__gGotham = PyGotham()

def __wrapMethod(name, wrapperName):
  firstLine = 'def ' + wrapperName + '(*args, **kwargs):\n'
  secondLine = '  return __gGotham.' + name + '(*args, **kwargs)\n'
  return firstLine + secondLine

# now wrap up the api in gGotham
for member in dir(__gGotham):
  # ignore standard stuff beginning '__'
  if member[0] != '_' and inspect.ismethod(getattr(__gGotham, member)):
    wrapperName = member[0].upper() + member[1:]
    exec __wrapMethod(member, wrapperName)

