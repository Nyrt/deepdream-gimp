import sys
import os
import textwrap
from shutil import copyfile

if os.getuid() != 0:
    admin_msg = "Please re-run this script as an Administrator (i.e. sudo) to install relevant files."
    print(textwrap.fill(admin_msg, 80))
    exit()
else:
    install_msg = "This installer will copy three files to your GIMP plug-ins folder. Do you wish to continue [Y/n]? "
    user_response = raw_input(textwrap.fill(install_msg, 80))
    if user_response.lower() not in ["yes", "y"]:
        print("Installation cancelled.")
        exit()


opsys = sys.platform
print(opsys)

dir1 = ""
dir2 = ""

if opsys == "linux" or opsys == "linux2" or opsys == "linux3":
    # using linux os
    dir1 = "/usr/lib/gimp/2.0/plug-ins/"
    dir2 = ""
elif opsys == "darwin":
    # using mac os
    dir1 = "/Applications/GIMP.app/Contents/Resources/lib/gimp/2.0/plug-ins/"
    dir2 = "/Library/Application Support/GIMP/2.8/plug-ins/"
elif opsys == "win32" or opsys == "cygwin":
    # using windows
    dir1 = ""
    dir2 = ""
else:
    # os not compatible with gimp
    dir1 = ""

dirDest = dir1

if os.path.isdir(dir1) and os.path.isdir(dir2):
    if len(os.listdir(dir2)) > len(os.listdir(dir1)):
        dirDest = dir2
else:
    if os.path.isdir(dir2):
        dirDest = dir2

print(dirDest)

dirSrc = os.getcwd() + "/gimp"
print(dirSrc)

files = os.listdir(dirSrc)
for f in files:
    print(f[-4:])
    if f[-4:] == ".txt":
        copyfile(dirSrc + "/" + f, dirDest + f)
