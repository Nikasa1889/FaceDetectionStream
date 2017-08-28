"""
Run command to push to git
"""
from subprocess import Popen, PIPE


message = "test deploy script"

p = Popen(["git", "pull"])
p.communicate()

p = Popen(["git", "add", "--all"])
p.communicate()

p = Popen(["git", "commit", "-m", message])
p.communicate()

p = Popen(["git", "push", "--force", "origin", "HEAD:master"])
p.communicate()
