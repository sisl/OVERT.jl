import os
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
# set up logging for DAgger

def setup_logging(log_dir, algo, env, novice, expert, DR):
	tabular_log_file = os.path.join(log_dir, "progress.csv")
	text_log_file = os.path.join(log_dir, "debug.log")
	params_log_file = os.path.join(log_dir, "params.json")
	snapshot_mode="last"
	snapshot_gap=1
	logger.add_text_output(text_log_file)
	logger.add_tabular_output(tabular_log_file)
	prev_snapshot_dir = logger.get_snapshot_dir()
	prev_mode = logger.get_snapshot_mode()
	logger.set_snapshot_dir(log_dir)
	logger.set_snapshot_mode(snapshot_mode)
	logger.set_snapshot_gap(snapshot_gap)
	print("Finished setting up logging.")

	# log some stuff
	logger.log("Created algorithm object of type: %s", type(algo))
	logger.log("env of type: %s" % type(env))
	logger.log("novice of type: %s" % type(novice))
	logger.log("expert of type: %s" % type(expert))
	logger.log("decision_rule of type: %s" % type(DR))
	logger.log("DAgger beta decay: %s" % DR.beta_decay)
	logger.log("numtrajs per epoch/itr: %s" % algo.numtrajs)
	logger.log("n_iter: %s" % algo.n_itr)
	logger.log("max path length: %s" % algo.max_path_length)
	logger.log("Optimizer info - ")
	logger.log("Optimizer of type: %s" % type(algo.optimizer))
	if type(algo.optimizer) == FirstOrderOptimizer:
	    logger.log("Optimizer of class: %s" % type(algo.optimizer._tf_optimizer))
	    logger.log("optimizer learning rate: %s" % algo.optimizer._tf_optimizer._lr)
	    logger.log("optimizer max epochs: %s" % algo.optimizer._max_epochs)
	    logger.log("optimizer batch size: %s" % algo.optimizer._batch_size)
	elif type(algo.optimizer) == PenaltyLbfgsOptimizer:
	    logger.log("initial_penalty %s" % algo.optimizer._initial_penalty)
	    logger.log("max_opt_itr %s" % algo.optimizer._max_opt_itr)
	    logger.log("max_penalty %s" % algo.optimizer._max_penalty)

	return True