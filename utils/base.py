import abc

class LogHelper (abc.ABC):

    def on_batch_end(self, outputs):
        pass

    def on_epoch_end(self, outputs):
        pass
    @abc.abstractmethod
    def check_pipeline(self):
        pass
