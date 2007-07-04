#!/usr/bin/env python
import sys
import os
import re
import tempfile
import shutil

def usage():
  pass

if len(sys.argv) < 2:
  print 'Error: No input file specified.'
  usage()
  exit()

# open the file
filename = sys.argv[1]
basename = os.path.basename(filename)
basename = basename.split('.')[0]
infile = open(filename)

# copy each line without a main(...)
shaderString = infile.read()

# replace "main(.*)" with "virtual Spectrum evaluate(const DifferentialGeometry &dg) const"
pattern = re.compile('main\((.*)\)')

try:
  replaceMe = 'main(' + pattern.search(shaderString).groups()[0] + ')'
  shaderString =  shaderString.replace(replaceMe,
                                       'virtual ScatteringFunction *evaluate(const DifferentialGeometry &dg) const')
except:
  print 'Error: No main function defined.'
  exit()

# standard includes
compileMe  = '''
#include "Lambertian.h"
#include "Material.h"
'''
compileMe += 'class %s : public Material\n' % basename
compileMe += '{\n'
compileMe += 'virtual const char *getName(void) const{ return "%s";}\n' % basename
compileMe += shaderString
compileMe += '};\n'

# create shared library export
compileMe += 'extern "C" Material * createMaterial(void)\n'
compileMe += '{\n'
compileMe += '  //std::cerr << "createMaterial(): Entered." << std::endl;\n'
compileMe += '  return new %s();\n' % basename
compileMe += '}\n'

# create temp files
dir = tempfile.mkdtemp()

cpppath = dir + '/' + basename + '.cpp'
makefile = '''
includes = ['/home/jared/dev/src', '/home/jared/dev/src/gotham/shading']
libs = ['shading']
libpath = ['.']
env = Environment(CPPPATH = includes, LIBS = libs, LIBPATH = libpath, SHLIBPREFIX = '')
sources = ['%s']
env.SharedLibrary('%s', sources)''' % (cpppath, basename)

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

