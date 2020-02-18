import tensorflow as tf # 1.0
import numpy as np
from tensorflow.python.framework import graph_util
import os

# write tf network to file
def write_graphdef(sess, output):
    output_graph_def = graph_util.convert_variables_to_constants(
            sess, # sess used to retrieve weights
            sess.graph.as_graph_def(), # graph def used to retrieve nodes
            [output.op.name] # output node name used to select useful nodes
            )

    cwd = os.getcwd()
    if not os.path.exists(os.path.join(cwd, 'tfpb/')):
        os.mkdir(os.path.join(cwd, 'tfpb/'))
    f_id = str(int(np.round(np.random.rand()*5000)))
    output_graph_filename = "tfpb/graph_def_"+f_id+".pb"
    with tf.gfile.GFile(output_graph_filename, "w") as f:
        f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_filename