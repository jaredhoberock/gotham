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
    result = ['/EHsc', '/MD', '/DWIN32']
  elif os.name == 'posix':
    #result = ['-O3', '-fPIC', '-Wall']
    result = ['-O3', '-fPIC']
  return result

def getDebugCPPFLAGS():
  result = []
  if os.name == 'nt':
    result = ['/EHsc', '/MDd', '/DWIN32']
  elif os.name == 'posix':
    result = ['-fPIC', '-g', '-Wall']
  return result;

def getLibraryPaths():
  result = []
  # figure out the absolute path of this file
  thisFile = inspect.getabsfile(getIncludes)
  if os.name == 'nt':
    windowsLib = os.path.join(os.path.dirname(thisFile), 'windows/lib')
    # XXX scons does not locate all the msvc 8 libs correctly
    sdkLib = "c:/Program Files/Microsoft SDKs/Windows/v6.0A/Lib"
    result = [windowsLib, sdkLib]
  return result

def getIncludes():
  result = []
  # figure out the absolute path of this file
  thisFile = inspect.getabsfile(getIncludes)
  # include the dependencies directory
  includeMe0 = os.path.join(os.path.dirname(thisFile), 'dependencies')
  # include the gotham include directory
  includeMe1 = os.path.join(os.path.dirname(thisFile), 'include')
  # include the gotham include/detail directory
  includeMe2 = os.path.join(os.path.dirname(thisFile), 'include/detail')
  if os.name == 'nt':
    windowsInclude = os.path.join(os.path.dirname(thisFile), 'windows/include')
    # XXX scons does not locate all the msvc 8 includes correctly
    sdkInclude = "c:/Program Files/Microsoft SDKs/Windows/v6.0A/Include"
    result = [includeMe0, includeMe1, includeMe2,
              windowsInclude,
              sdkInclude,
              os.path.join(windowsInclude, 'OpenEXR'),
              os.path.join(windowsInclude, 'python2.5')]
  elif os.name == 'posix':
    result = [includeMe0, includeMe1, includeMe2, '/usr/include/python2.5',
              '/usr/include/OpenEXR']
    result += ['/usr/include/qt4',
              '/usr/include/qt4/Qt',    '/usr/include/qt4/QtCore',
              '/usr/include/qt4/QtGui', '/usr/include/qt4/QtXml', '/usr/include/qt4/QtOpenGL']
  return result

def GothamReleaseEnvironment(env):
  env.Append(CPPPATH = getIncludes())
  env.Append(LIBPATH = getLibraryPaths())
  env.Append(CPPFLAGS = getReleaseCPPFLAGS())

def GothamDebugEnvironment(env):
  env.Append(CPPPATH = getIncludes())
  env.Append(LIBPATH = getLibraryPaths())
  env.Append(CPPFLAGS = getDebugCPPFLAGS())

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

