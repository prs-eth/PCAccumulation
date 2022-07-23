from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanTracker(object):
    """
    Internal state management for each tracklet
    https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
    """
    count=0
    def __init__(self, observation, info, configs):
        """
        Initialise the tracker from first hit 
        We will initialise the initial state, state transition model, observation model, state covariance matrix, and process uncertainty matrix
        kf.H:   [obs_dim, state_dim]    process model   
        kf.P:   [state_dim, state_dim]  state covariance matrix
        kf.R:   [obs_dim, obs_dim] measurement uncertainty/noise
        kf.Q:   [state_dim, state_dim] process noise


        Input:
            observation:    [obs_dim,1]
            info:           a dictionary describing the observation
        """
        self.state_dim, self.obs_dim = configs['state_dim'], configs['obs_dim']
        self.kf = KalmanFilter(dim_x = configs['state_dim'], dim_z = configs['obs_dim'])
        self.id = KalmanTracker.count   # unique id for each tracker
        KalmanTracker.count+=1

        if(self.state_dim == 6 and self.obs_dim == 3):
            # initial the initial state
            self.kf.x[:3,0] = observation.flatten()
            self.kf.x[3,0] = configs['vx']
            
            # state transition matrix [state_dim, state_dim]
            self.kf.F = np.array([[1,0,0,1,0,0],     
                                  [0,1,0,0,1,0],
                                  [0,0,1,0,0,1],
                                  [0,0,0,1,0,0],
                                  [0,0,0,0,1,0],
                                  [0,0,0,0,0,1]])    
            # measurement function    [obs_dim, state_dim]
            self.kf.H = np.array([[1,0,0,0,0,0],
                                  [0,1,0,0,0,0],
                                  [0,0,1,0,0,0]])    

            # state covariance matrix, here we follow AB3DMOT and give high variance to unobserved velocity
            self.kf.P[3:,3:] *= configs['velocity_uncertainty']
            self.kf.P[:3,:3] *= configs['pos_uncertainty']

            # process uncertainty matrix [state_dim, state_dim]
            self.kf.Q[3:,3:] *= configs['process_uncertainty']
        elif(self.state_dim==4 and self.obs_dim==2):
            # initial the initial state
            self.kf.x[:2,0] = observation.flatten()
            self.kf.x[2,0] = configs['vx']
            # state transition matrix [state_dim, state_dim]
            self.kf.F = np.array([[1,0,1,0],     
                                  [0,1,0,1],
                                  [0,0,1,0],
                                  [0,0,0,1]
                                  ])    
            # measurement function    [obs_dim, state_dim]
            self.kf.H = np.array([[1,0,0,0],
                                  [0,1,0,0]])    

            # state covariance matrix, here we follow AB3DMOT and give high variance to unobserved velocity
            self.kf.P[2:,2:] *= configs['velocity_uncertainty']
            self.kf.P[:2,:2] *= configs['pos_uncertainty']

            # process uncertainty matrix [state_dim, state_dim]
            self.kf.Q[2:,2:] *= configs['process_uncertainty']
        else:
            raise NotImplementedError


        # keep tracking the stats of the track
        self.hits = 1  # number of total hits
        self.hits_streak = 1 # number of continued hits
        self.hits_streak_since_initialisation = 1
        self.age = 0  # number of total prediction
        self.track_lost = False
        self.frames_since_last_update = 0  # 
        self.history = []  # store the tracking history, it records: [frame_id, instance_id, frame_score]
        self.history.append(info)

    @property
    def prediction_uncertainty(self):
        """
        Compute the uncertainty of the current prediction
        Here self.kf.R is an identity matrix
        """
        S = np.matmul(np.matmul(self.kf.H, self.kf.P), self.kf.H.T) + self.kf.R
        return S


    def predict(self):
        """
        Advance the state vector and return the predicted geometric blob
        Always run predict to synchronise the timestamp
        Return:
            prediction:     [state_dim, 1]
        """
        self.kf.predict()
        prediction = self.kf.x

        self.age += 1 
        if(self.frames_since_last_update != 0):  # when we lost track 
            self.hits_streak = 0
            self.track_lost = True
        self.frames_since_last_update += 1

        return prediction


    def update(self, observation, info):
        """
        Update the state vector with associated geometric blob
        Input:
            observation:   [obs_dim, 1]
            info:           a dictionary has [score, instance_id, frame_id]
        """
        self.kf.update(observation)
        self.history.append(info)

        self.hits += 1


        self.hits_streak += 1 
        if(not self.track_lost):
            self.hits_streak_since_initialisation += 1 
        self.frames_since_last_update = 0
        


class MultiClusterTrackingManager(object):
    """
    For each frame, we associate each cluster to one old tracklet or initialise a new tracklet
    """
    def __init__(self, configs):
        self.configs = configs
        self.state_dim, self.obs_dim = configs['state_dim'], configs['obs_dim']
        self.max_age = configs['max_age']     # if kf.frames_since_last_update == max_age, then we kill this tracker
        self.min_hits = configs['min_hits']    # if kf.hits == min_hits, then we give birth to this tracker
        self.match_algo = configs['match_algorithm']
        self.mahalanobis_threshold =configs['mahalanobis_threshold']    # if the matching cost is above this threshold, then it's considered as unmatched
        self.trackers = []
    
    def clear(self):
        self.trackers = []

    def format_tracking_result(self,tracker):
        track_history = tracker.history
        track_score = np.mean([obs['score'] for obs in track_history])
        track_length = len(track_history)
        instance_ids = [obs['instance_id'] for obs in track_history]

        result = {
            "tracker_id": tracker.id,
            "track_history": tracker.history,
            'track_score': track_score,
            'track_length': track_length,
            'instance_ids': instance_ids,
            'state': tracker.kf.x
        }
        return result


    def _compute_cost(self, obs, track_preds, S):
        """
        Compute Mahalanobis distance between all the observations and all the predictions
        Input:
            obs:            [N, 3]
            track_preds:    [M, 3]
            S:              [M, 3, 3]
        Return:
            cost_matrix:    N, M
        """
        n_obs = obs.shape[0]
        n_tracks = track_preds.shape[0]
        cost_matrix = np.zeros((n_obs, n_tracks))
        cost_matrix_wo_S = np.zeros((n_obs, n_tracks))
        
        flag = n_obs * n_tracks > 0
        if(flag):
            inv_S = np.linalg.inv(S)
            for i in range(n_obs):
                for j in range(n_tracks):
                    diff = (obs[i] - track_preds[j])[:,None]  #[3, 1]
                    cost_matrix[i,j] = np.sqrt(np.matmul(np.matmul(diff.T, inv_S[j]),diff))[0][0]
                    cost_matrix_wo_S[i,j] = np.sqrt(np.matmul(diff.T,diff))[0][0]

        return cost_matrix, cost_matrix_wo_S

    def _associate(self, cost_matrix):
        """
        Run data association
        1. get initial matching results from greedy algorithm
        2. filter out matches with high matching cost by thresholding 
        3. retain unmatched observations and tracks

        Input:
            cost_matrix:    [n_obs, n_tracks]
        Output:
            real_matches:       [p, 2]
            unmatched_obs:      [m]
            unmatched_tracks:   [n]
        """
        n_obs, n_tracks = cost_matrix.shape
        flag = n_obs * n_tracks

        if(flag == 0):
            real_matches = np.zeros((0,2),dtype=np.int32)
            unmatched_obs = np.arange(n_obs)
            unmatched_tracks = np.arange(n_tracks)
            return real_matches, unmatched_obs, unmatched_tracks


        initial_matches, real_matches = [],[]
        unmatched_obs, unmatched_tracks = [],[]
        if(self.match_algo == 'greedy'): # greedy matching 
            distance_1d = cost_matrix.reshape(-1)
            index_1d = np.argsort(distance_1d)
            index_2d = np.stack([index_1d // n_tracks, index_1d % n_tracks], axis=1)
            obs_id_matches_to_track_id = [-1] * n_obs
            track_id_matches_to_obs_id = [-1] * n_tracks
            for sort_i in range(index_2d.shape[0]):
                detection_id = int(index_2d[sort_i][0])
                tracking_id = int(index_2d[sort_i][1])
                if track_id_matches_to_obs_id[tracking_id] == -1 and obs_id_matches_to_track_id[detection_id] == -1:
                    track_id_matches_to_obs_id[tracking_id] = detection_id
                    obs_id_matches_to_track_id[detection_id] = tracking_id
                    initial_matches.append([detection_id, tracking_id])
        elif(self.match_algo == 'hungarian'):
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            initial_matches = np.stack((row_ind, col_ind), axis=1)
        else:
            raise NotImplementedError

        # 2. filter out bad matches
        for eachmatch in initial_matches:
            score = cost_matrix[eachmatch[0], eachmatch[1]]
            if(score < self.mahalanobis_threshold):
                real_matches.append(eachmatch)   
        if(len(real_matches)>0):     
            real_matches = np.array(real_matches)
        else:
            real_matches = np.zeros((0,2),dtype=np.int32)

        # 3. get unmatched observations and trackers
        for idx in range(n_obs):
            if(idx not in real_matches[:,0]):
                unmatched_obs.append(idx)
        for idx in range(n_tracks):
            if(idx not in real_matches[:,1]):
                unmatched_tracks.append(idx)
        
        return real_matches, unmatched_obs, unmatched_tracks


    def _manage_track_birth_and_death(self, real_matches, unmatched_obs, unmatched_tracks, obs, infos):
        """
        Manage the birth and death of tracklets after association
        1. update matched trackers with assigned detections
        2. initialise new tracks for unmatched observations
        3. remove dead trackers
        4. write trajectory to json file
        
        Input:
            real_matches:       [p, 2]  (obs_id, track_id)
            unmatched_obs:      [q]
            unmatched_tracks:   [t]
        Return:
            information of the dead trackers
        """

        # 1. udpate matched trackers
        n_matches = real_matches.shape[0]
        for idx in range(n_matches):
            self.trackers[real_matches[idx,1]].update(obs[real_matches[idx,0]],infos[real_matches[idx,0]])

        # 2. initialise new tracks
        n_new_obs = len(unmatched_obs)
        for idx in range(n_new_obs):
            self.trackers.append(KalmanTracker(obs[unmatched_obs[idx]],infos[unmatched_obs[idx]],self.configs))
        
        # 3. remove dead trackers
        retired_trackers = []
        n_trackers = len(self.trackers)
        for tracker in reversed(self.trackers):
            n_trackers -=1
            if(tracker.frames_since_last_update >= self.max_age):
                self.trackers.pop(n_trackers)
                retired_trackers.append(self.format_tracking_result(tracker))
        
        return retired_trackers

    def update(self,obs, infos):
        """
        Update the trackers from new observations
        1. make predictions from existing trackers
        2. run data association modules
        3. initiliase new trackers for unmatched observations, update matched trackers and kill dead trackers

        Input:
            obs:   [N, 3], here the obs could also be empty
            infos:  list of N dictionary, each dictionary has [score, instance_id, frame_id]
        """
        # each existing tracker makes predictions
        tracks_preds = []
        to_del = []
        for idx, eachtracker in enumerate(self.trackers):
            prediction = eachtracker.predict().flatten()
            
            if(np.any(np.isnan(prediction))): # delete bad trackers
                to_del.append(idx)
            else:
                tracks_preds.append(prediction)
        tracks_preds = np.array(tracks_preds)

        if(tracks_preds.shape[0]):
            tracks_preds = tracks_preds[:,:self.obs_dim]


        # remove bad trackers
        for t in reversed(to_del):
            print('bad tracker detected')
            self.trackers.pop(t)

        # compute matching cost
        trks_S = [tracker.prediction_uncertainty for tracker in self.trackers]
        trks_S = np.array(trks_S)
        trk_obs_info ={
            'trk': [tracker.history[0]['instance_id'] for tracker in self.trackers],
            'obs': [info['instance_id'] for info in infos]
        }
        cost_matrix, cost_matrix_wo_S = self._compute_cost(obs, tracks_preds, trks_S)

        # run data association
        real_matches, unmatched_obs, unmatched_tracks = self._associate(cost_matrix)

        # manage the birth and death of tracks
        dead_trackers = self._manage_track_birth_and_death(real_matches, unmatched_obs, unmatched_tracks, obs, infos)
        return dead_trackers