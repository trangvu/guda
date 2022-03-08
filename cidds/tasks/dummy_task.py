from fairseq.tasks import FairseqTask, register_task


@register_task("dummy_task")
class DummyTask(FairseqTask):
    """
    Dummy Task to build model since Fairseq requires a task to build and load model
    """

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        src_dict = kwargs['src_dict']
        tgt_dict = kwargs['tgt_dict']
        return cls(args, src_dict, tgt_dict)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.tgt_dict