import tensorflow as tf

class TDOptimizer(tf.optimizers.Optimizer):
    """Optimizer that implements the TD lambda eligibility traces
    The update weight is performed as
    trace = trace * lambda + grad
    weight = weight - trace * learning_rate"""
    def __init__(self,
            name = 'TDOptimizer',
            learning_rate=1e-3,
            tdlambda=0.8,
            **kvargs):
        """Creates a TD lambda eligibility trace optimizer"""
        super(TDOptimizer, self).__init__(name=name,
            gradient_aggregator=None,
            gradient_transformers=None,
             *kvargs)
        self._set_hyper("learning_rate", kvargs.get("lr", learning_rate))
        self._set_hyper("tdlambda", tdlambda)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "trace")

    def get_config(self):
        config = super(TDOptimizer, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "tdlambda": self._serialize_hyperparameter("tdlambda"),
        })
        return config

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device = var.device
        var_dtype = var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

        lr_t = coefficients["lr_t"]
        trace_var = self.get_slot(var, 'trace')
        tdlambda = self._get_hyper('tdlambda')

        new_trace_var = trace_var * tdlambda + grad
        new_var = var - new_trace_var * lr_t

        trace_var.assign(new_trace_var)
        var.assign(new_var)
    
    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError
