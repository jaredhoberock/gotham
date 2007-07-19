#!/usr/bin/env python

# TODO
# mangle scattering and emission parameter names so they don't conflict
# during compilation OR just make both sets of names visible to both
# functions?

import sys
import os
import re
import tempfile
import shutil

def parseParameters(parameters):
  result = []
  # first check for void parameters
  voidParameters = re.compile('^\s*$|^\s*void\s*$')
  if voidParameters.search(parameters) != None:
    return result
  # match parameters
  # split on ','
  declarations = parameters.split(',')
  for decl in declarations:
    try:
      (type, identifier) = tuple(decl.split())
      # add a declaration to the result
      result.append((type, identifier))
    except:
      print 'Error: Expected a declaration.'
      exit()
  return result

def parseFunction(name, source):
  # find the parameters of name():
  slurpUntilEndParens = '[^\)]*'
  parametersPattern = re.compile(('%s\((' % name) + slurpUntilEndParens + ')\)')
  parametersMatch = parametersPattern.search(source)
  parameters = parametersMatch.groups()[0]
  parametersList = parseParameters(parameters)
  # find the body of the function
  body = ''
  searchMeForBody = source[parametersMatch.end(0):]
  # strip off leading whitespace
  searchMeForBody = searchMeForBody.lstrip()
  if searchMeForBody[0] != '{':
    raise ValueError, "Error: expected '{'."
  depth = 1
  for c in searchMeForBody[1:]:
    if c == '{':
      depth += 1
    elif c == '}':
      depth -= 1
    if depth == 0:
      break
    body += c
  if depth != 0:
    raise ValueError, "Error: EOF reached before end of %s." % name
  return (parametersList, body)


def compile(filename):
  globalVariables = '''
const Point &P = dg.getPoint();
const Normal &N = dg.getNormal();
const ParametricCoordinates &UV = dg.getParametricCoordinates();
const Vector3 &dpdu = dg.getPointPartials()[0];
const Vector3 &dpdv = dg.getPointPartials()[1];
  '''
  
  # open the file
  basename = os.path.basename(filename)
  basename = basename.split('.')[0]
  infile = open(filename)
  
  # copy each line without a main(...)
  shaderString = infile.read()
  
  # try to parse 'scattering'
  scatteringString = ''
  scatteringParametersList = []
  try:
    (scatteringParametersList, scatteringBody) = parseFunction('scattering', shaderString)
    
    # rebuild a shader string
    scatteringString = '''
virtual ScatteringDistributionFunction *evaluateScattering(const DifferentialGeometry &dg) const
{
  ScatteringDistributionFunction *F = 0;
  '''
    scatteringString += globalVariables
    scatteringString += scatteringBody
    scatteringString += '''
  return F;
}
'''
  except:
    pass
  
  # try to parse 'emission'
  emissionString = ''
  emissionParametersList = []
  try:
    (emissionParametersList, emissionBody) = parseFunction('emission', shaderString)
    emissionString = '''
 virtual ScatteringDistributionFunction *evaluateEmission(const DifferentialGeometry &dg) const
 {
   ScatteringDistributionFunction *E = 0;
 '''
    emissionString += globalVariables
    emissionString += emissionBody
    emissionString += '''
   return E;
 }
 virtual bool isEmitter(void) const
 {
   return true;
 }
 '''
  except:
    pass
  
  # try to parse 'sensor'
  sensorString = ''
  sensorParametersList = []
  try:
    (sensorParametersList, sensorBody) = parseFunction('sensor', shaderString)
    sensorString = '''
virtual ScatteringDistributionFunction *evaluateSensor(const DifferentialGeometry &dg) const
{
  ScatteringDistributionFunction *S = 0;
'''
    sensorString += globalVariables
    sensorString += sensorBody
    sensorString += '''
  return S;
}
virtual bool isSensor(void) const
{
  return true;
}
'''
  except:
    pass
  
  
  # XXX we need to put the preamble outside of the class definition
  
  # standard includes
  compileMe = '''
#include "Lambertian.h"
#include "Material.h"
#include "HemisphericalEmission.h"
#include "PerspectiveSensor.h"
#include <spectrum/Spectrum.h>
#include "../geometry/Point.h"
#include "../geometry/Vector.h"
'''
  
  # XXX put the preamble just before the class definition
  #compileMe += preamble
  
  compileMe += 'class %s : public Material\n' % basename
  compileMe += '{\n'
  compileMe += 'public:\n'
  compileMe += 'virtual const char *getName(void) const{ return "%s";}\n' % basename
  compileMe += scatteringString
  compileMe += emissionString
  compileMe += sensorString
  # add parameters
  for param in scatteringParametersList:
    compileMe += param[0] + ' ' + param[1] + ';\n'
    compileMe += 'void set_%s(const %s &v){%s = v;}\n' % (param[1], param[0], param[1])
  for param in emissionParametersList:
    compileMe += param[0] + ' ' + param[1] + ';\n'
    compileMe += 'void set_%s(const %s &v){%s = v;}\n' % (param[1], param[0], param[1])
  for param in sensorParametersList:
    compileMe += param[0] + ' ' + param[1] + ';\n'
    compileMe += 'void set_%s(const %s &v){%s = v;}\n' % (param[1], param[0], param[1])
  compileMe += '};\n'
  
  # create shared library export
  compileMe += 'extern "C" Material * createMaterial(void)\n'
  compileMe += '{\n'
  compileMe += '  return new %s();\n' % basename
  compileMe += '}\n'
  
  compileMe += '''
#include <boost/python.hpp>
#include <boost/python/manage_new_object.hpp>
#include <boost/python/return_value_policy.hpp>
using namespace boost::python;
BOOST_PYTHON_MODULE(%s)
{
  def("createMaterial", createMaterial, return_value_policy<manage_new_object>());
  class_<Vector>("Vector", init<float,float,float>());
  class_<Point>("Point", init<float,float,float>());
  class_<Spectrum>("Spectrum", init<float,float,float>());
  class_<%s, bases<Material> >("%s")
  ''' % (basename,basename,basename)
  for param in scatteringParametersList:
    compileMe += '    .def_readwrite("%s", &%s::%s)\n' % (param[1], basename, param[1])
    compileMe += '    .def("set_%s", &%s::set_%s)\n' % (param[1], basename, param[1])
  for param in emissionParametersList:
    compileMe += '    .def_readwrite("%s", &%s::%s)\n' % (param[1], basename, param[1])
    compileMe += '    .def("set_%s", &%s::set_%s)\n' % (param[1], basename, param[1])
  for param in sensorParametersList:
    compileMe += '    .def_readwrite("%s", &%s::%s)\n' % (param[1], basename, param[1])
    compileMe += '    .def("set_%s", &%s::set_%s)\n' % (param[1], basename, param[1])
  compileMe += '    ;\n'
  
  # close the module
  compileMe += '}\n'
  
  #print compileMe
  
  # create temp files
  dir = tempfile.mkdtemp()
  
  cpppath = dir + '/' + basename + '.cpp'
  makefile = '''
includes = ['/home/jared/dev/src', '/home/jared/dev/src/gotham/shading', '/usr/include/python2.5']
libs = ['shading', 'geometry', 'boost_python']
libpath = ['../geometry','../shading']
env = Environment(CPPPATH = includes,
                  CPPFLAGS = '-O3',
                  LIBS = libs,
                  LIBPATH = libpath,
                  SHLIBPREFIX = '',
                  tools = ['default', 'intelc'])
sources = ['%s']
env.SharedLibrary('%s', sources)
''' % (cpppath, basename)
  
  cppfile = open(cpppath, 'w')
  cppfile.write(compileMe)
  cppfile.close()
  
  sconstruct = open(dir + '/' + 'SConstruct', 'w')
  sconstruct.write(makefile)
  sconstruct.close()
  
  # call scons
  command = 'scons -Q -f %s' % dir + '/' + 'SConstruct' 
  os.system(command)
  
  # kill the tempfiles
  shutil.rmtree(dir, True)
  

def usage():
  pass

if len(sys.argv) < 2:
  print 'Error: No input file specified.'
  usage()
  exit()

for file in sys.argv[1:]:
  compile(file)

