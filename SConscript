import os
import glob

Import('env')

# add dependencies

# linking to libstdc++ first worksaround
# a bug which occurs on some 64b Linux systems
# see http://wiki.fifengine.de/Segfault_in_cxa_allocate_exception
if os.name == 'posix':
  env.Append(LIBS = ['stdc++'])

# with msvc, boost figures out which libraries to link
if os.name == 'posix':
  env.Append(LIBS = ['boost_python'])
  env.Append(LIBS = ['boost_thread'])

if os.name == 'nt':
  env.Append(LIBS = ['python25'])
else:
  env.Append(LIBS = ['python2.5'])

# image format libraries
env.Append(LIBS = ['jpeg'])
env.Append(LIBS = ['png'])

# openexr libraries
env.Append(LIBS = ['Iex'])
env.Append(LIBS = ['IlmImf'])
env.Append(LIBS = ['Imath'])
env.Append(LIBS = ['Half'])
env.Append(LIBS = ['z'])

# graphics libraries
if os.name == 'nt':
  env.Append(LIBS = ['glut32'])
  env.Append(LIBS = ['opengl32'])
  env.Append(LIBS = ['glu32'])
  env.Append(LIBS = ['glew32'])
else:
  env.Append(LIBS = ['glut'])
  env.Append(LIBS = ['GL'])
  env.Append(LIBS = ['GLU'])
  env.Append(LIBS = ['GLEW'])

## qtviewer libraries
#env.Append(LIBS = ['QtCore'])
#env.Append(LIBS = ['QtXml'])
#env.Append(LIBS = ['QtGui'])
#env.Append(LIBS = ['QtOpenGL'])
#env.Append(LIBS = ['qglviewer'])

# aggregate shared objects from each child directory
subdirectories = ['geometry',
                  'importance',
                  'mutators',
                  'path',
                  'primitives',
                  'rasterizables',
                  'records',
                  'renderers',
                  'shading',
                  'surfaces',
                  'api']
sharedObjects = []
for dir in subdirectories:
  sharedObjects += SConscript('%s/SConscript' % dir, exports={'env':env})

# use the 'lib-' prefix on all platforms
env['SHLIBPREFIX'] = 'lib'

# accomodate windows 
if os.name == 'nt':
  # on windows, python dlls must have the suffix, '.pyd'
  env['SHLIBSUFFIX'] = '.pyd'

# create a shared library
gothamLib = env.SharedLibrary('gotham', sharedObjects)

# accomodate windows
if os.name == 'nt':
  # we must explicitly embed the "manifest" after building the dll
  # from http://www.scons.org/wiki/EmbedManifestIntoTarget 
  # XXX fix this issue where mt.exe cannot be found without an absolute path
  env.AddPostAction(gothamLib, '"C:/Program Files/Microsoft SDKs/Windows/v6.0A/bin/mt.exe" -nologo -manifest ${TARGET}.manifest -outputresource:$TARGET;2')
  # delete the manifest file
  env.AddPostAction(gothamLib, 'del.exe ${TARGET}.manifest')
  # delete the import library - we don't need it
  env.AddPostAction(gothamLib, 'del.exe ${TARGET.filebase}.lib')
  # delete the .exp library
  env.AddPostAction(gothamLib, 'del.exe ${TARGET.filebase}.exp')

# install the distribution
homedir = '.'
try:
  homedir = os.environ["GOTHAMHOME"]
except:
  print "Warning: $GOTHAMHOME not defined! Gotham could not be installed."

dir = homedir + '/lib'
Default(env.Install(dir, source = gothamLib))
Default(env.Install(dir, source = 'api/api.py'))
Default(env.Install(dir, source = 'api/objtogoth.py'))
Default(env.Install(dir, source = 'api/wavefront.py'))
dir = homedir + '/bin'
Default(env.Install(dir, source = 'api/pbslc'))
Default(env.Install(dir, source = 'api/gotham'))

# accomodate windows 
if os.name == 'nt':
  # install dummy batch files for the compiler and renderer
  Default(env.Install(dir, source = 'api/pbslc.bat'))
  Default(env.Install(dir, source = 'api/gotham.bat'))

  # install all *.dlls in windows/lib
  pattern = os.path.join(os.getcwd(), "windows/lib/*.dll")
  for fullpath in glob.glob(pattern):
    Default(env.Install(dir, source = fullpath))

  # install Python headers in include/detail
  Default(env.Install(os.path.join(homedir, 'include/detail'), source = 'windows/include/python2.5'))

  # install boost headers in include/detail
  # XXX all that is actually needed here is Boost.Python and dependencies
  Default(env.Install(os.path.join(homedir, 'include/detail'), source = 'windows/include/boost'))

  # install Python library in lib
  Default(env.Install(os.path.join(homedir, 'lib'), source = 'windows/lib/python25.lib'))

  # install Boost.Python library in lib
  Default(env.Install(os.path.join(homedir, 'lib'), source = 'windows/lib/boost_python-vc90-mt-1_36.lib'))

# build imgstat
env.Program('imgstat', 'api/imgstat.cpp')
dir = homedir + '/bin'
Default(env.Install(dir, source = 'imgstat'))

# build imgclean
env.Program('imgclean', 'api/imgclean.cpp')
dir = homedir + '/bin'
Default(env.Install(dir, source = 'imgclean'))

# build imgaccum
env.Program('imgaccum', 'api/imgaccum.cpp')
dir = homedir + '/bin'
Default(env.Install(dir, source = 'imgaccum'))

# build test if it exists
if os.path.exists('test.cpp'):
  env.Program('test', 'test.cpp')

