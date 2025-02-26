'''ConeState.py

This file contains a class for using prior color estimates of cones to inform
and recolor new/incoming cone color estimates. The current implementation
uses ICP to determine correspondences between cones that have been seen (in state)
and new incoming cones. 

Then, each cone in the state has a counter for the
number of timesteps/prediction iterations they have been in the state
and another counter for the number of times the lidar coloring part of
the pipeline predicted them as yellow. Then, for cones that obtain
correspondences, their counts are updated based on the pipeline's color
predictions at that iteration. Cones in state that are not seen in the new
set of cones are discarded, and cones that haven't found an associated cone
in the existing state are added as a new cone to the state.
'''

# perc22a imports
from perc22a.predictors.utils.cones import Cones
from perc22a.utils.globe.MotionInfo import MotionInfo

from perc22a.utils.Timer import Timer

import perc22a.predictors.utils.icp as icp

# general imports
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=3)


class ConeState:
    '''Maintains a state of the cones that are being produced by various
    pipelines. In order to maintain additional metadata, the cones are 
    represented as a numpy array in the `self.cones_state_arr` attribute
    
    The structure of a state array for a cone is described as follows.
    
    The state array representation is going to be as follows (N x 7) array
        The 7 columns are defined as follows
            0. x position of cone
            1. y position of cone
            2. z position of cone
            3. # times cone has been seen as a yellow cone
            4. # times cone has been received by .update() call
            5. # times cone has not been seen (after being initially seen)
            6. # times cone has not been consecutively seen
    '''


    # TODO: determine what is the best merging policy?
    #   1. do it like counting and take the color with the highest count
    #   2. do we assume that the existing state is correct and ignore the old state
    #       in this case, what do if error in new incoming cone?

    # TODO: could accumulate transformations to get more cones in state
    # TODO: maybe just use ICP to inform decision about seed for lidar coloring?

    # TODO: weaknesses
    #   1. if a cone disappears, it's counts get totally refreshed which isn't good
    #   2. if looking at side, then new cones coming in might propogate wrong color
    #   - need uncorrelated cones to at least get colors from corresponded colors

    def __init__(self):
        # current state of cones with attached motion information
        self.cones_state_arr = None
        self.state_mi = None

        # indices to represent the cone array
        self.YELLOW_COUNT_IDX = 3
        self.SEEN_COUNT_IDX = 4
        self.NO_CORR_COUNT_IDX = 5
        self.CONSECUTIVE_NO_CORR_COUNT_IDX = 6

        # search correspondences over 1m radius
        self.icp_max_correspondence_dist = 1.25
        self.icp_max_iters = 30

        # for debugging
        self.timer = Timer()

        pass

    def _cones_to_state_arr(self, cones: Cones):
        '''Converts a new cones object to a state array representation'''
        blue, yellow, _ = cones.to_numpy()

        b0, b1 = np.zeros((blue.shape[0], 1)), np.ones((blue.shape[0], 1))
        y0, y1 = np.zeros((yellow.shape[0], 1)), np.ones((yellow.shape[0], 1))

        # x, y, z, yellow-count, seen-count, missed-count, consecutively-missed-count
        blue = np.concatenate([blue, b0, b1, b0, b0], axis=1)
        yellow = np.concatenate([yellow, y1, y1, y0, y0], axis=1)

        return np.concatenate([blue, yellow], axis=0)
       
    def _transform_and_corr(self, src_points, src_mi, dest_points, dest_mi):
        '''use own icp implementation that is singled threaded and pure NumPy

        This function will initially use GPS pose information stored in
        MotionInfo objects to create an initial transformation to go from
        source to destination points.

        Then, ICP is used to fine-tune transformation between two
        sets of points and determine the correspondences between them.

        Arguments:
            - src_points: point array to transform to destination
            - src_mi: MotionInfo associated with src_points
            - dest_points: state array of cones from current timestep
            - dest_mi: MotionInfo associated with destination points
        
        Returns:
            - transformed_prev_state: prior state of cones but with x and y
                positions to be updated w.r.t. car at current timestep using ICP
            - corr: (K x 2) integer nparray of indices representing
                corresponences between prev and curr cone state arrays
                note: (K <= min(# num prev cones, # num curr cones))
        ''' 
        assert(src_points.ndim == 2 and src_points.shape[1] == 2)
        assert(dest_points.ndim == 2 and dest_points.shape[1] == 2)

        # first transform src to destination using GPS MotionInfo as an guess
        n_src_points = src_points.shape[0]
        src_points_3d = np.concatenate([src_points, np.zeros((n_src_points, 1))], axis=1)
        src_points = src_mi.model_motion_to(src_points_3d, dest_mi)[:, :2]

        # perform icp
        corr, T, corr_dists, iters, transformed_src = icp.icp(
            src_points, dest_points,
            init_pose=None,
            max_iterations=self.icp_max_iters,
            max_corr_dist=self.icp_max_correspondence_dist
        )

        # icp.debug_correspondences(src_points, dest_points, corr)
        # icp.debug_correspondences(transformed_src, dest_points, corr)
        
        # update the old positions of the cones using the transformation
        # useful for updating position of uncorrelated cones
        return transformed_src, corr
    
    def _filter_state(self, cones_state_arr):
        '''Filters out stale cones based on criteria that cone has been seen
        fewer times than it has been not correlated
        '''

        # TODO: at start of execution, car is standing still so seen_count
        # accumulates to be extremely large, might need to cap it
        seen_count = cones_state_arr[:, self.SEEN_COUNT_IDX]
        no_corr_count = cones_state_arr[:, self.NO_CORR_COUNT_IDX]
        is_fresh = seen_count - no_corr_count > 0

        fresh_state_arr = cones_state_arr[is_fresh, :]
        return fresh_state_arr
    
    def _filter_state_window(self, cones_state_arr, n=1):
        '''Filters out stale cones based on criteria that cone has not bee
        consecutively seen for n timesteps
        '''
        cons_no_corr_count = cones_state_arr[:, self.CONSECUTIVE_NO_CORR_COUNT_IDX]
        is_fresh = cons_no_corr_count < n 

        fresh_state_arr = cones_state_arr[is_fresh, :]
        return fresh_state_arr
        
    
    def _update_state(self, cones_state_arr, new_cone_arr, correspondences):

        # NOTE: this is where the primary update policy is implemented
        # and how merging past estimates works with merging current estimates

        # 3 groups
        # in correspondence set (update positions and update counts)
        # cone in cone state not in correspondence set (remove from state)
        # cone in new cones (add into state, initialize counts)

        state_corr = correspondences[:, 0]
        new_corr = correspondences[:, 1]

        # 1. update cones found in the correspondence set
        corr_state_cones = cones_state_arr[state_corr, :]
        corr_new_cones = new_cone_arr[new_corr, :]

        # for each corr state cone, update it's positions and counts
        corr_state_cones[:, :3] = corr_new_cones[:, :3]
        corr_state_cones[:, self.YELLOW_COUNT_IDX] += corr_new_cones[:, self.YELLOW_COUNT_IDX]
        corr_state_cones[:, self.SEEN_COUNT_IDX] += 1
        corr_state_cones[:, self.CONSECUTIVE_NO_CORR_COUNT_IDX] = 0

        # 2. update metadata old cones not found in correspondence set
        state_uncorr_mask = np.ones(cones_state_arr.shape[0], dtype=bool)
        state_uncorr_mask[state_corr] = False

        uncorr_state_cones = cones_state_arr[state_uncorr_mask, :]
        uncorr_state_cones[:, self.NO_CORR_COUNT_IDX] += 1
        uncorr_state_cones[:, self.CONSECUTIVE_NO_CORR_COUNT_IDX] += 1

        # 3. add new cones not found in the correspondence set
        new_uncorr_mask = np.ones(new_cone_arr.shape[0], dtype=bool)
        new_uncorr_mask[new_corr] = False

        uncorr_new_cones = new_cone_arr[new_uncorr_mask, :]

        # now merge correlated and uncorrelated new cones for updated state
        new_state_arr = np.concatenate([corr_state_cones, uncorr_state_cones, uncorr_new_cones], axis=0)

        # filter stale cones
        new_state_arr = self._filter_state_window(new_state_arr)
        
        return new_state_arr


    def _state_to_svm_cones(self, cones_state_arr):

        # get indices for blue and yellow cones based on predictions
        # TODO: need better tie-breaking scheme?
        yellow_prob = cones_state_arr[:, 3] / cones_state_arr[:, 4]
        blue_idxs = np.where(yellow_prob < 0.5)
        yellow_idxs = np.where(yellow_prob >= 0.5)

        blue_cones_arr = cones_state_arr[blue_idxs][:, :3]
        yellow_cones_arr = cones_state_arr[yellow_idxs][:, :3]
        orange_cones_arr = np.zeros((0, 3))

        return Cones.from_numpy(blue_cones_arr, yellow_cones_arr, orange_cones_arr)
    

    def update(self, new_cones: Cones, new_mi: MotionInfo):
        if len(new_cones) == 0:
            # TODO: is this the best behavior, should use prior state if possible?
            return new_cones

        if self.cones_state_arr is None or self.cones_state_arr.shape[0] <= 1:
            self.cones_state_arr = self._cones_to_state_arr(new_cones)
            self.state_mi = new_mi
            return new_cones
        
        # convert cones into a point cloud of cones
        new_cone_pc_arr = self._cones_to_state_arr(new_cones)

        # use icp to get correspondences and set cone state w.r.t curr car pos
        updated_state_pos, corr = self._transform_and_corr(
            self.cones_state_arr[:, :2],
            self.state_mi, 
            new_cone_pc_arr[:, :2],
            new_mi
        )  
        if corr is None:
            # if unable to find correspondences
            # return current cones without integrating into state
            # TODO: must determine if this is the appropriate behavior
            # typically occurs when many cones disappear or no cones available
            return new_cones
        
        self.cones_state_arr[:, :2] = updated_state_pos

        # update the overall cone state and MotionInfo associated with it
        self.cones_state_arr = self._update_state(
            self.cones_state_arr,
            new_cone_pc_arr,
            corr
        )
        self.state_mi = new_mi

        # convert existing state into a Cones object
        # print(np.round(self.cones_state_arr, 3))
        cones = self._state_to_svm_cones(self.cones_state_arr)

        return cones
    
    def get_svm_cones(self):
        return None