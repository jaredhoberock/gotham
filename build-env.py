#!/usr/bin/env python
import os
import inspect

def getTools():
  result = []
  if os.name == 'nt':
    result = ['default', 'msvc']
  elif os.name == 'posix':
    #result = ['default', 'intelc']
    # XXX BUG boost_random produces negative numbers in
    #         icc optimized code (-O2 and above)
    # see http://lists.boost.org/Archives/boost/2007/07/124815.php for details
    result = ['default']
  return result

def getReleaseCPPFLAGS():
  result = []
  if os.name == 'nt':
    result = ['/EHsc', '/MD']
  elif os.name == 'posix':
    #result = ['-O3', '-fPIC', '-Wall']
    result = ['-O3', '-fPIC']
  return result

def getDebugCPPFLAGS():
  result = []
  if os.name == 'nt':
    result = ['/EHsc', '/MDd']
  elif os.name == 'posix':
    result = ['-fPIC', '-g', '-Wall']
  return result;

def getIncludes():
  result = []
  # figure out the absolute path of this file
  thisFile = inspect.getabsfile(getIncludes)
  # include the dependencies directory
  includeMe0 = os.path.join(os.path.dirname(thisFile), 'dependencies')
  # include the gotham include directory
  includeMe1 = os.path.join(os.path.dirname(thisFile), 'include')
  if os.name == 'nt':
    result = [includeMe0, includeMe1, 'c:/Python25/include',
              'c:/dev/include/OpenEXR']
  elif os.name == 'posix':
    result = [includeMe0, includeMe1, '/usr/include/python2.5',
              '/usr/include/OpenEXR']
    result += ['/usr/include/qt4',
              '/usr/include/qt4/Qt',    '/usr/include/qt4/QtCore',
              '/usr/include/qt4/QtGui', '/usr/include/qt4/QtXml', '/usr/include/qt4/QtOpenGL']
  return result

def GothamReleaseEnvironment(env):
  env.Append(CPPPATH = getIncludes())
  env.Append(CPPFLAGS = getReleaseCPPFLAGS())

def GothamDebugEnvironment(env):
  env.Append(CPPPATH = getIncludes(),
             CPPFLAGS = getDebugCPPFLAGS())

def GothamEnvironment():
  mode = 'release'
  env = Environment(tools = getTools())

  if ARGUMENTS.get('mode'):
    mode = ARGUMENTS['mode']
  if mode == 'release':
    GothamReleaseEnvironment(env)
  elif mode == 'debug':
    GothamDebugEnvironment(env)
  else:
    raise ValueError, 'Unknown target mode "%s".' % mode
  return env

