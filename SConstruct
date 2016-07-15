"""SCons script for buildling the CubeN package (library and test programs).
"""

import os
import re
import platform

# Define environment configuration
name = 'cuben'
env = Environment(CPPPATH=['inc','src'])
ispc = platform.system().lower() == 'windows'
if ispc:
    env['CPPFLAGS'] = ['/EHsc', '/nologo']
    epicPath = os.environ['APPDATA'] + os.path.sep + 'EPiC' + os.path.sep
else:
    epicPath = os.path.expanduser('~') + os.path.sep + '.local' + os.path.sep + 'EPiC' + os.path.sep
env['CPPPATH'].append(epicPath + 'inc')
    
# Compile .C/CPP files in src/ into object files
obj = []
for f in os.listdir('src'):
    if re.search('\.c(pp)?$', f):
        obj.append(env.Object('src'+os.path.sep+f))

# Link object files into a static library
lib = Library('lib' + os.path.sep + name, obj)

# Now build executables from contents of test folder files 'test_[...].c(pp)'
for f in os.listdir('test'):
    if re.search('^test_(.+)\.c(pp)?$', f):
        binName = re.sub('\.c(pp)?$', '', f)
        env.Program('test'+os.path.sep+binName, ['test'+os.path.sep+f,lib])
