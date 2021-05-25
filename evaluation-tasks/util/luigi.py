"""
Common Luigi classes and functions for evaluation tasks
"""

import os
import luigi


def ensure_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class WorkTask(luigi.Task):
    """
    We assume following conventions:
        * Each luigi Task will have a name property:
            {classname}
            or
            {classname}-{task parameters}
            depending upon what your want the name to be.
            (TODO: Since we always use {classname}, just
            make this constant?)
        * The "output" of each task is a touch'ed file,
        indicating that the task is done. Each .run()
        method should end with this command:
            `_workdir/done-{name}`
            * Optionally, working output of each task will go into:
            `_workdir/{name}`
    Downstream dependencies should be cautious of automatically
    removing the working output, unless they are sure they are the
    only downstream dependency of a particular task (i.e. no
    triangular dependencies).
    """

    @property
    def name(self):
        ...
        # return type(self).__name__

    def output(self):
        return luigi.LocalTarget("_workdir/done-%s" % self.name)

    @property
    def workdir(self):
        d = "_workdir/%s/" % self.name
        ensure_dir(d)
        return d
