#!BPY

"""
Name: 'Gotham'
Blender: 237
Group: 'Export'
Tooltip: '.py exporter'
"""

import Blender
from Blender import Material, Mathutils

# for os.path
import os

# globals
gTurbidity = 3
gSunGain = [0.02, 0.02, 0.02]

def writeMatrix(indent, matrix, out):
  out.write(indent + "g.multMatrix([")
  for i in range(4):
    for j in range(4):
      if i == 3 and j == 3:
        out.write("%f " % matrix[j][i])
      else:
        out.write("%f, " % matrix[j][i])
  out.write("])\n")

def writeMesh(mesh, out):
  # write out a wavefront obj to a file-like object and pass this
  # along to the wavefront code
  # write positions
  lines = []
  for vert in mesh.verts:
    lines.append('v %f %f %f\n' % (vert.co.x, vert.co.y, vert.co.z))

  # write vertex normals
  numVertexNormals = 0
  for vert in mesh.verts:
    lines.append('vn %f %f %f\n' % (vert.no.x, vert.no.y, vert.no.z))
    numVertexNormals = numVertexNormals + 1

  # write face normals
  lines.append('# face normals:\n')
  for face in mesh.faces:
    if not face.smooth:
      lines.append('vn %f %f %f\n' % (face.no[0], face.no[1], face.no[2]))

  # write faces
  faceIndex = 0
  for face in mesh.faces:
    line = 'f'
    if face.smooth:
      for vert in face.v:
        line += ' %i//%i' % (vert.index + 1, vert.index + 1)
    else:
      for vert in face.v:
        line += ' %i//%i' % (vert.index + 1, numVertexNormals + faceIndex + 1)
    line += '\n'
    lines.append(line)
    faceIndex = faceIndex + 1

  # make it act like a file
  def readlines(): return lines
  class foo: pass
  file = foo()
  file.readlines = readlines

  import objtogoth
  (vertices, parms, indices) = objtogoth.objtogoth(file)
  print >> out, 'g.mesh(', vertices, ",", parms, ",", indices, ')'

def writeMaterial(materials, out):
  if len(materials) > 0:
    if materials[0].emit > 0:
      # this is a light
      # XXX fix this
      #     try to fish out the power/gain whatever
      print >> out, "g.material('light', 'power', (1,1,1))"
    else:
      mat = materials[0]
      if (mat.mode & Material.Modes.RAYTRANSP) and (mat.mode & Material.Modes.RAYMIRROR):
        # write glass
        Kr = (mat.rayMirr, mat.rayMirr, mat.rayMirr)
        Kt = (mat.R, mat.G, mat.B)
        eta = mat.IOR
        print >> out, "g.material('perfectglass', 'Kr', (%f, %f, %f), 'Kt', (%f, %f, %f), 'eta', %f )" % (Kr[0], Kr[1], Kr[2], Kt[0], Kt[1], Kt[2], eta)
      elif mat.mode & Material.Modes.RAYTRANSP:
        # write transmission
        Kt = (mat.R, mat.G, mat.B)
        etat = mat.IOR
        print >> out, "g.material('speculartransmission', 'Kt', (%f, %f, %f), 'etai', 1.0, 'etat', %f)" % (Kt[0], Kt[1], Kt[2], etat)
      elif mat.mode & Material.Modes.RAYMIRROR:
        # write mirror
        # XXX we could pass along IOR
        Kr = (mat.rayMirr, mat.rayMirr, mat.rayMirr)
        print >> out, "g.material('mirror', 'Kr', (%f, %f, %f))" % (Kr[0], Kr[1], Kr[2])
      else:
        eta = mat.IOR
        specular = mat.getSpec()
        exponent = mat.hard * 10
        Kd = (mat.R, mat.G, mat.B)
        Kr = (specular * mat.specR, specular * mat.specG, specular * mat.specB)
        if specular == 1.0:
          # write phong reflection
          print >> out, "g.material('phongreflection', 'Kr', (%f, %f, %f), 'eta', %f, 'exponent', %f)" % (Kr[0], Kr[1], Kr[2], eta, exponent)
        elif specular > 0.0001:
          # write uber (diffuse + phong)
          print >> out, "g.material('constantuber', 'Kd', (%f, %f, %f), 'Ks', (%f, %f, %f), 'exponent', %f)" % (Kd[0], Kd[1], Kd[2], Kr[0], Kr[1], Kr[2], exponent)
        else:
          # write matte
          print >> out, "g.material('matte', 'Kd', (%f, %f, %f))" % Kd
  else:
    print >> out, "g.material('matte', 'Kd', (0.8,0.8,0.8))"

def writePrimitive(indent, prim, out):
  print >> out, '# --- %s ---' % prim.getName()
  print >> out, "g.attribute('name', '%s')" % prim.getName()
  if prim.getType() == "Mesh":
    # push matrix
    print >> out, 'g.pushMatrix()'
    # write matrix here
    writeMatrix(indent, prim.getMatrix(), out)
    # write shader here
    writeMaterial(prim.getData().materials, out)
    # now write the mesh
    writeMesh(prim.getData(), out)
    print >> out, 'g.popMatrix()'
  else:
    pass
  print >> out, '\n'

def writeCamera(camera, w, h, out):
  matrix = camera.getMatrix()
  eye = (matrix[3][0], matrix[3][1], matrix[3][2])
  look = matrix[2]
  up = (matrix[1][0], matrix[1][1], matrix[1][2])

  # subtract look rather than add to get the point we're looking at
  # this is a consequence of right-handed coordinate systems
  lookAt = (eye[0] - look[0],
            eye[1] - look[1],
            eye[2] - look[2])

  aspect = float(w) / h

  print >> out, '# --- camera ---'
  print >> out, 'g.pushMatrix()'
  print >> out, 'g.lookAt(', eye, ',', lookAt, ',', up, ')'
  print >> out, 'g.camera(%f, 60.0, 0.01)' % (aspect)
  print >> out, 'g.popMatrix()'
  print >> out, '\n'

def write(filename):
  indent = ""
  out = file(filename, "w")

  # get the base of the filename
  # chuck off the file extention
  filenameNoExtension = '.'.join(filename.split('.')[0:-1])
  sceneFilename = filenameNoExtension + '.scene.py'
  sceneOut = file(sceneFilename, "w")

  scn = Blender.Scene.GetCurrent()

  ## output lights first
  #for object in scn.getChildren():
  #  if object.getType() == "Lamp":
  #    if object.data.getType() == 1:
  #      # sun is type 1
  #      invMat = Mathutils.Matrix(object.getInverseMatrix())
  #      writeSkylight(indent, invMat[0][2], invMat[1][2], invMat[2][2], gTurbidity, out)
  #    elif object.data.getType() == 4:
  #      # directional is type 4
  #      invMat = Mathutils.Matrix(object.getInverseMatrix())
  #      e = object.data.getEnergy()
  #      c = object.data.getCol()
  #      writeDirectionalLight(indent, invMat[0][2], invMat[1][2], invMat[2][2], e*c[0], e*c[1], e*c[2], out)

  # output objects to the scene file
  for prim in scn.objects:
    writePrimitive(indent, prim, sceneOut)

  # output preamble
  print >> out, "#!/usr/bin/env python"
  print >> out, "import api"
  print >> out, "g = api.Gotham2()\n"

  # output code to slurp in the scene file
  sceneRelpath = './' + os.path.basename(sceneFilename)
  print >> out, "lines = open('%s').readlines()" % sceneRelpath
  print >> out, "numLines = len(lines)"
  print >> out, "fivePercent = numLines / 20"
  print >> out, "lineNumber = 0"
  print >> out, "print 'Parsing scene...'"
  print >> out, "for line in lines:"
  print >> out, "  if lineNumber % fivePercent == 0:"
  print >> out, "    print 'Progress: ', 100 * float(lineNumber)/numLines, '%' "
  print >> out, "  exec line"
  print >> out, "  lineNumber += 1\n"

  w = scn.getRenderingContext().imageSizeX()
  h = scn.getRenderingContext().imageSizeY()

  # output camera to the frontend file
  camera = scn.objects.camera
  if camera:
    writeCamera(camera, w, h, out)

  imageName = filenameNoExtension + ".exr"
  imageName = './' + os.path.basename(imageName)

  # output code to set up the render
  print >> out, "g.attribute('path::maxlength', '20')"
  print >> out, "g.attribute('path::sampler', 'simpleforwardrussianroulette')"
  print >> out, "g.attribute('path::russianroulette::function', 'kelemen')"
  print >> out, "g.attribute('path::russianroulette::continueprobability', '0.95')"
  print >> out, "g.attribute('renderer::algorithm', 'metropolis')"
  print >> out, "g.attribute('renderer::targetrays', '50000000')"
  print >> out, "g.attribute('record::outfile', '%s')" % imageName

  print >> out, '\n'
  print >> out, "g.render((%d,%d), 1)" % (w,h)

Blender.Window.FileSelector(write, "Export")

