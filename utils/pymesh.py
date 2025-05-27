import pymeshfix

infile = "/Users/niyaz/Documents/dreamgaussian/logs/3dguy_test _backup/.obj"
outfile = "output_mesh.obj"
# Read mesh from infile and output cleaned mesh to outfile
pymeshfix.clean_from_file(infile, outfile)
