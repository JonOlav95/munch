import yaml

f = open("../parameters.yml", "r")
FLAGS = yaml.safe_load(f)
f.close()
