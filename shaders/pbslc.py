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
    elif c == '}': depth -= 1
    if depth == 0:
      break
    body += c
  if depth != 0:
    raise ValueError, "Error: EOF reached before end of %s." % name
  return (parametersList, body)


def makeShaderParameterCode(parameterType, parameterName):
  # declare the parameter
  result = '%s %s;\n' % (parameterType, parameterName)
  # make a setter method for it
  if parameterType == "Vector" or parameterType == "Point" or parameterType == "Spectrum":
    result += 'void set_%s(const float x, const float y, const float z){%s = %s(x,y,z);}\n' % (parameterName, parameterName, parameterType)
  elif parameterType == "float":
    result += 'void set_%s(const %s &v){%s = v;}\n' % (parameterName, parameterType, parameterName)
  else:
    print "Error: Unknown type %s for parameter %s." % (parameterType, parameterName)
    exit()
  return result


def compile(filename):
  globalVariables = '''
const Point &P = dg.getPoint();
const Normal &N = dg.getNormal();
const Vector &T = dg.getTangent();
const Vector &B = dg.getBinormal();
const ParametricCoordinates &UV = dg.getParametricCoordinates();
const Vector3 &dpdu = dg.getPointPartials()[0];
const Vector3 &dpdv = dg.getPointPartials()[1];
      float area = dg.getSurface()->getSurfaceArea();
      float invArea = dg.getSurface()->getInverseSurfaceArea();
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
#include "stdshader.h"
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
    compileMe += makeShaderParameterCode(param[0], param[1])
  for param in emissionParametersList:
    compileMe += makeShaderParameterCode(param[0], param[1])
  for param in sensorParametersList:
    compileMe += makeShaderParameterCode(param[0], param[1])
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
  class_<%s, bases<Material> >("%s")
  ''' % (basename,basename,basename)
  for param in scatteringParametersList:
    compileMe += '    .def("set_%s", &%s::set_%s)\n' % (param[1], basename, param[1])
  for param in emissionParametersList:
    compileMe += '    .def("set_%s", &%s::set_%s)\n' % (param[1], basename, param[1])
  for param in sensorParametersList:
    compileMe += '    .def("set_%s", &%s::set_%s)\n' % (param[1], basename, param[1])
  compileMe += '    ;\n'
  
  # close the module
  compileMe += '}\n'
  
  #print compileMe
  
  # create temp files
  dir = tempfile.mkdtemp()

  cpppath = dir + '/' + basename + '.cpp'

  # replace all \ with /
  # XXX for some reason scons messes up if we don't do this
  #     \ in the dir are interpreted as escape characters
  cpppath = cpppath.replace('\\', '/')

  # XXX fix this include ugliness
  #     ideally, there's a canonical installation of gotham somewhere
  #     with an include directory. point to that.
  makefile = '''
import os
if os.name == 'posix':
  includes = ['/home/jared/dev/src', '/home/jared/dev/src/gotham/shading', '/usr/include/python2.5']
elif os.name == 'nt':
  includes = ['c:/dev/src', 'c:/dev/include', 'c:/dev/src/gotham/shading', 'c:/Python25/include']
if os.name == 'posix':
  # remember gotham doesn't have the 'lib-' prefix
  libs = [File('/home/jared/dev/src/gotham/api/gotham.so'), 'boost_python']
elif os.name == 'nt':
  # on windows, we need to link to the import library
  libs = ['gotham']
# fix these path issues
if os.name == 'posix':
  libpath = ['/home/jared/dev/src/gotham/api']
elif os.name == 'nt':
  libpath = ['c:/dev/src/gotham/api', 'c:/dev/lib', 'c:/Python25/libs']
env = None
if os.name == 'posix':
  env = Environment(CPPPATH = includes,
                    CPPFLAGS = '-O3',
                    LIBS = libs,
                    LIBPATH = libpath,
                    SHLIBPREFIX = '',
                    tools = ['default', 'intelc'])
elif os.name == 'nt':
  env = Environment(CPPPATH = includes,
                    CPPFLAGS = ['/Ox', '/EHsc', '/MD', '/DIMPORTDLL=1'],
                    LIBS = libs,
                    LIBPATH = libpath,
                    SHLIBPREFIX = '',
                    SHLIBSUFFIX = '.pyd',
                    WINDOWS_INSERT_MANIFEST = True)
sources = ['%s']
result = env.SharedLibrary('%s', sources)
if os.name == 'nt':
  # we must explicitly embed the "manifest" after building the dll
  # from http://www.scons.org/wiki/EmbedManifestIntoTarget 
  env.AddPostAction(result, 'mt.exe -nologo -manifest ${TARGET}.manifest -outputresource:$TARGET;2')
  # delete the manifest file
  env.AddPostAction(result, 'del.exe ${TARGET}.manifest')
  # delete the import library - we don't need it
  env.AddPostAction(result, 'del.exe ${TARGET.filebase}.lib')
  # delete the .exp library
  env.AddPostAction(result, 'del.exe ${TARGET.filebase}.exp')
''' % (cpppath, basename)
  
  cppfile = open(cpppath, 'w')
  cppfile.write(compileMe)
  cppfile.close()
  
  sconstruct = open(dir + '/' + 'SConstruct', 'w')
  sconstruct.write(makefile)
  sconstruct.close()
  
  # call scons
  # XXX cygwin has a fit if you call it scons when its name is scons.bat
  #     fix this nonsense
  if os.name == 'posix':
    command = 'scons -Q -f %s' % dir + '/' + 'SConstruct' 
  elif os.name == 'nt':
    command = 'c:/Python25/scons.bat -Q -f %s' % dir + '/' + 'SConstruct' 
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

