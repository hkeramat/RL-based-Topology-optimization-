import os
import sys
import pygmsh
import meshio
from shape_generator import *

# less than 3 point does not generate a closed shape
if (len(sys.argv) != 2):
    print('more than two points is required')
    sys.exit(0)

# retriev filename
filename = sys.argv[1]
if (not os.path.isfile(filename)):
    print('Input file does not exist')
    quit()

# Generate shape
shape = Shape()
shape.read_csv(filename)
shape.generate(ccws=True)
shape.mesh()
shape.generate_image(plot_pts=True)
