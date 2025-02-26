from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

from perc22a.mergers.BaseMerger import BaseMerger
from perc22a.mergers.PipelineType import PipelineType

from perc22a.data.utils.dataloader import DataLoader

from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.utils.Timer import Timer

from perc22a.predictors.utils.ConeState import ConeState

from perc22a.svm.SVM import SVM

import numpy as np

def main():
    lp = LidarPredictor()
    t = Timer()
    vis = Vis2D()

    # create merger
    merger = BaseMerger(required_pipelines=[], debug=True, zed_dist_limit=20, lidar_dist_limit=20)

    state = ConeState()

    dl = DataLoader("perc22a/data/raw/tt-4-18-fourth")
    svm = SVM()

    for i in range(71, len(dl)):

        t.start("time")
        t.start("\tlidar")
        # perform prediction
        cones = lp.predict(dl[i])
        t.end("\tlidar")

        # merge the cones together from other pipelines
        # TODO: is this necessary now?
        t.start("\tmerge")
        merger.add(cones, PipelineType.LIDAR)
        cones = merger.merge() 
        merger.reset()
        t.end("\tmerge")

        # color the cones
        t.start("\tcolor")
        cones = svm.recolor(cones)
        t.end("\tcolor")

        t.start("\tstate")
        cones = state.update(cones)
        t.end("\tstate")

        # convert cones to SVM midline points
        t.start("\tmidline")
        midline_points = svm.cones_to_midline(cones)
        t.end("\tmidline")
        t.end("time")

        vis.set_cones(cones)
        if len(midline_points) > 0:
            vis.set_points(midline_points)
            vis.update()        



if __name__ == "__main__":
    main()
