#!/usr/bin/env python
import sys
import wavefront

def splitTriangles(points, triangles):
  newPoints = []
  newTriangles = []
  index = 0
  for (v0,v1,v2) in triangles:
    newPoints.append(points[v0[0]])
    newPoints.append(points[v1[0]])
    newPoints.append(points[v2[0]])
    newV0 = (index,     -1, -1)
    newV1 = (index + 1, -1, -1)
    newV2 = (index + 2, -1, -1)
    newTriangles.append([newV0, newV1, newV2])
    index += 3
  return (newPoints, newTriangles)

def splitTrianglesWithParametrics(points, uvs, triangles):
  newPoints = []
  newUvs = []
  newTriangles = []
  index = 0
  for (v0,v1,v2) in triangles:
    newPoints.append(points[v0[0]])
    newPoints.append(points[v1[0]])
    newPoints.append(points[v2[0]])
    newUvs.append(uvs[v0[1]])
    newUvs.append(uvs[v1[1]])
    newUvs.append(uvs[v2[1]])
    newV0 = (index,     index,     -1)
    newV1 = (index + 1, index + 1, -1)
    newV2 = (index + 2, index + 2, -1)
    newTriangles.append([newV0, newV1, newV2])
    index += 3
  return (newPoints, newUvs, newTriangles)

def objtogoth(filename):
  (points, uvs, normals, polygons) = wavefront.readMesh(filename)
  triangles = wavefront.triangulate(polygons)
  if uvs != []:
    (points, uvs, triangles) = splitTrianglesWithParametrics(points, uvs, triangles)
  else:
    (points, triangles) = splitTriangles(points, triangles)

  vertices = []
  indices = []
  parms = []
  # flatten the points list
  for p in points:
    vertices.extend(p)
  # flatten the uv list
  for uv in uvs:
    parms.extend(uv)
  # flatten the triangle list
  for tri in triangles:
    indices.append(tri[0][0])
    indices.append(tri[1][0])
    indices.append(tri[2][0])
  return (vertices, parms, indices)

def objtogothWithMaterials(filename):
  results = []
  (points, uvs, normals, materials, primitives) = wavefront.readMeshes(filename)
  (materials, primitives) = wavefront.partitionByMaterials(points, uvs, normals, materials, primitives)
  for (material, points, uvs, normals, polygons) in primitives:
    triangles = wavefront.triangulate(polygons)
    if uvs != []:
      (points, uvs, triangles) = splitTrianglesWithParametrics(points, uvs, triangles)
    else:
      (points, triangles) = splitTriangles(points, triangles)

    vertices = []
    indices = []
    parms = []
    # flatten the points list
    for p in points:
      vertices.extend(p)
    # flatten the uv list
    for uv in uvs:
      parms.extend(uv)
    # flatten the triangle list
    for tri in triangles:
      indices.append(tri[0][0])
      indices.append(tri[1][0])
      indices.append(tri[2][0])

    # convert from wavefront material to gotham material
    try:
      wavefrontParms = materials[material]
    except:
      print 'looking up material', material
      raise ValueError, materials

    exponent = 10000
    if wavefrontParms.has_key('Ns'):
      exponent = wavefrontParms['Ns']

    etai     = 1.0

    etat = 1000
    if wavefrontParms.has_key('Ni'):
      etat     = wavefrontParms['Ni']

    Kd = (0,0,0)
    if wavefrontParms.has_key('Kd'):
      Kd       = wavefrontParms['Kd']

    Kt = (0,0,0)
    if wavefrontParms.has_key('Tf'):
      Kt = wavefrontParms['Tf']

    Ks = (0,0,0)
    if wavefrontParms.has_key('Kt'):
      Ks       = wavefrontParms['Ks']

    # figure out which scattering properties we have
    hasDiffuse = False
    hasTransmission = False
    hasSpecular = False

    if Kd[0] > 0 or Kd[1] > 0 or Kd[2] > 0:
      hasDiffuse = True
    if Kt[0] > 0 or Kt[1] > 0 or Kt[2] > 0:
      hasTransmission = True
    if Ks[0] > 0 or Ks[1] > 0 or Ks[2] > 0:
      hasSpecular = True

    gothamName = ''
    gothamParms = {}
    if not hasDiffuse and not hasTransmission and hasSpecular:
      gothamName = 'phongreflection'
      gothamParms['Kr'] = Ks
      gothamParms['eta'] = etat
      gothamParms['exponent'] = exponent
    elif not hasDiffuse and hasTransmission and not hasSpecular:
      gothamName = 'speculartransmission'
      gothamParms['Kt'] = Kt
      gothamParms['etai'] = etai
      gothamParms['etat'] = etat
    elif not hasDiffuse and hasTransmission and hasSpecular:
      if eta > 1:
        gothamName = 'perfectglass'
        gothamParms['eta'] = eta
      else:
        gothamName = 'thinglass'
      gothamParms['Kt'] = Kt
      gothamParms['Kr'] = Ks
    elif hasDiffuse and not hasTransmission and not hasSpecular:
      gothamName = 'matte'
      gothamParms['Kd'] = Kd
    elif hasDiffuse and not hasTransmission and hasSpecular:
      gothamName = 'constantuber'
      gothamParms['Kd'] = Kd
      gothamParms['Ks'] = Ks
      gothamParms['exponent'] = exponent
    elif hasDiffuse and hasTransmission and not hasSpecular:
      print "Warning: objtogoth::objtogothWithMaterials(): material '%s' with diffuse and transmission has no Gotham analogue." % material
      gothamName = 'matte'
      gothamParms['Kd'] = Kd
    elif hasDiffuse and hasTransmission and hasSpecular:
      print "Warning: objtogoth::objtogothWithMaterials(): material '%s' with diffuse, transmission, and specular has no Gotham analogue." % material
      gothamName = 'matte'
      gothamParms['Kd'] = Kd
    else:
      print "Warning: objtogoth::objtogothWithMaterials(): material '%s' has no scattering properties. Object will appear black." % material
      print materials[material]
      print 'hasDiffuse:', hasDiffuse
      print 'hasTransmission:', hasDiffuse
      print 'hasSpecular:', hasDiffuse
      gothamName = 'matte'
      gothamParms['Kd'] = Kd

    # add to the list
    results.append(((gothamName, gothamParms), vertices, parms, indices))
  return results

# did we call this as a program?
if sys.argv[0] == './objtogoth.py':
  if len(sys.argv) < 2:
    print 'usage: objtogoth input.obj'
  else:
    (vertices, uvs, triangles) = objtogoth(sys.argv[1])
    # print a mesh call
    print 'g.mesh(',
    print vertices,
    print ',',
    print uvs,
    print ',',
    print triangles,
    print ')'
  
