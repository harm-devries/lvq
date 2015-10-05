import numpy
from blocks.serialization import secure_dump
from blocks.extensions import SimpleExtension
from blocks.extensions.monitoring import MonitoringExtension
from scipy.linalg import svd

class MomentumSwitchOff(SimpleExtension):
    def __init__(self, momentum, switch_off_at, **kwargs):
        self.iter = 0
        self.momentum = momentum
        self.switch_off_at = switch_off_at
        kwargs.setdefault("after_epoch", True)
        super(MomentumSwitchOff, self).__init__(**kwargs)
        
    def do(self, which_callback, *args):
        self.iter += 1
        if self.iter == self.switch_off_at:
            self.momentum.set_value(numpy.float32(0.0))

class LRDecay(SimpleExtension):
    def __init__(self, lr, decay_epochs=[], **kwargs):
        self.iter = 0
        self.decay_epochs = decay_epochs
        self.lr = lr
        kwargs.setdefault("after_epoch", True)
        super(LRDecay, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        self.iter += 1
        if self.iter in self.decay_epochs:

            self.lr.set_value(numpy.float32(self.lr.get_value()/10.0))
        
        
class NCScheduler(SimpleExtension):
    def __init__(self, lambd, end_val, first, num_epochs, **kwargs):
        self.iter = 0
        self.lambd = lambd
        self.begin_val = lambd.get_value()
        self.num_epochs = num_epochs
        self.first = first
        self.end_val = end_val
        kwargs.setdefault("after_epoch", True)
        super(NCScheduler, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        self.iter += 1.0
        if self.iter > self.first:
            val = self.begin_val + (self.end_val - self.begin_val)*(self.iter - self.first)/self.num_epochs
            self.main_loop.log.current_row['lambda'] = val
            self.lambd.set_value(numpy.float32(val))
        

class EarlyStopping(SimpleExtension):
    """Check if a log quantity has the minimum/maximum value so far,
    and early stops the experiment if the quantity has not been better
    since `patience` number of epochs. It also saves the best best model
    so far.
    Parameters
    ----------
    record_name : str
        The name of the record to track.
    patience : int
        The number of epochs to wait before early stopping.
    path : str
        The path where to save the best model.
    notification_name : str, optional
        The name for the record to be made in the log when the current
        value of the tracked quantity is the best so far. It not given,
        'record_name' plus "best_so_far" suffix is used.
    choose_best : callable, optional
        A function that takes the current value and the best so far
        and return the best of two. By default :func:`min`, which
        corresponds to tracking the minimum value.
    Attributes
    ----------
    best_name : str
        The name of the status record to keep the best value so far.
    notification_name : str
        The name used for the notification
    """
    def __init__(self, record_name, patience, path, notification_name=None,
                 choose_best=min, **kwargs):
        self.record_name = record_name
        if not notification_name:
            notification_name = record_name + "_best_so_far"
        self.notification_name = notification_name
        self.best_name = "best_" + record_name
        self.choose_best = choose_best
        self.counter = 0
        self.path = path
        self.patience = patience
        kwargs.setdefault("after_epoch", True)
        super(EarlyStopping, self).__init__(**kwargs)

    def _dump(self):
        try:
            self.main_loop.log.current_row['saved_best_to'] = self.path
            secure_dump(self.main_loop, self.path)
        except Exception:
            self.main_loop.log.current_row['saved_best_to'] = None
            raise

    def do(self, which_callback, *args):
        current_value = self.main_loop.log.current_row.get(self.record_name)
        if current_value is None:
            self.counter += 1
            return
        best_value = self.main_loop.status.get(self.best_name, None)
        if (best_value is None or
                (current_value != best_value and
                 self.choose_best(current_value, best_value) ==
                 current_value)):
            self.main_loop.status[self.best_name] = current_value
            self.main_loop.log.current_row[self.notification_name] = True
            self.counter = 0
            self._dump()
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.main_loop.log.current_row['training_finish_requested'] = True
        self.main_loop.log.current_row['patience'] = self.counter
