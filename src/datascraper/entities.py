import numpy as np
from sklearn.cluster import KMeans 
from sklearn.linear_model import LinearRegression

"""Entities involved in processed and unprocessed Video objects.

Unprocessed Videos are composed of a sequence of Frames. 
Frames are further composed of a Board and a Person, which track
the moonboard Holds and human pose / keypoints.
"""

L_BICEP = 0
L_FOREARM = 1
R_BICEP = 2
R_FOREARM = 3
L_QUAD = 4
L_CALF = 5
R_QUAD = 6 
R_CALF = 7

class Frame:
    def __init__(self, board, person):
        self.board = board 
        self.person = person

class Board:
    def __init__(self):
        # Number of moonboard rows
        self.hand_box_width = 18 
        # Number of moonboard columns
        self.hand_box_length = 11
        # Number of kickbox rows
        self.foot_box_width = 2 
        # Number of kickbox columns
        self.foot_box_length = 5
        self.hand_box_dimension = self.hand_box_width*self.hand_box_length
        self.foot_box_dimension = self.foot_box_width*self.foot_box_length
        # Stores the moonboard holds
        self.handholds = np.empty(self.hand_box_dimension,dtype=object)
        self.footholds = np.empty(self.foot_box_dimension,dtype=object)
        # Odd column consists of holds in each odd moonboard column
        self.odd_column = []
        # Stores slope and y-intercept of lines passing through odd-column holds
        self.odd_column_line_params = []

    def set_holds_from_video(self,holds):
        """Set the Board hand and foot holds detected from a video.

        Args:
            holds: a list of detected Hold objects

        Returns:
            None
        """
        list(map(self.set_handhold,holds))
        missing_holds = list(map(self.get_interpolated_missing_handhold, self.get_missing_handhold_ids()))
        list(map(self.set_handhold,missing_holds))

        self.set_foothold_lines()
        self.detect_footholds(holds)
        missing_footholds = list(map(self.get_interpolated_missing_foothold, self.get_missing_foothold_ids()))
        list(map(self.set_foothold,missing_footholds))
    
    def set_handhold(self,hold):
        """Set the Board hand holds detected from a video.

        Args:
            holds: a list of detected Hold objects

        Returns:
            None
        """
        if hold.is_hand():
            hold_index = hold.get_hold_id()
            self.handholds[hold_index] = hold

    def set_foothold(self,hold):
        """Set the Board foot holds detected from a video.

        Args:
            holds: a list of detected Hold objects

        Returns:
            None
        """
        if hold.is_hand() == False:
            hold_index = int(hold.name)
            self.footholds[hold_index] = hold

    def set_foothold_lines(self):
        """Set line of best fit parameters for the odd column holds used to 
           predict foot hold positions

        Args:
            None

        Returns:
            None
        """
        self.odd_column = list(map(self.get_handholds_in_column,[1,3,5,7,9]))
        self.odd_column_line_params = list(map(self.line_of_best_fit, self.odd_column)) 

    def detect_footholds(self, holds):
        """Detect and organize Board footholds detected from a video.

        Args:
            holds: a list of detected Hold objects

        Returns:
            None
        """
        foot_holds = np.array(list(filter(lambda x: x.is_hand() == False, holds)))
        bottom_row = foot_holds[self.cluster_holds_y(foot_holds) == 1]
        top_row = foot_holds[self.cluster_holds_y(foot_holds) == 0]
        for hold in bottom_row:
            hold_to_line_distances = np.array(list(map(hold.distance_to_hold_line, self.odd_column_line_params)))
            hold_column = hold_to_line_distances.argsort()[0]
            hold.name = str(hold_column + 5)
            self.footholds[hold_column + 5] = hold
        for hold in top_row:
            hold_to_line_distances = np.array(list(map(hold.distance_to_hold_line, self.odd_column_line_params)))
            hold_column = hold_to_line_distances.argsort()[0]
            hold.name = str(hold_column)
            self.footholds[hold_column] = hold

    def cluster_holds_y(self,holds):
        """Classify detected holds into two groups based on their y-coordinate.

        Args:
            holds: a list of detected Hold objects

        Returns:
            k_means.labels_: a list of 0 and 1s, classifying the hold
        """
        hold_centers = np.array(list(map(lambda x: x.center, holds)))[:,1]
        x = hold_centers.reshape(-1,1)
        k_means = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(x)
        return k_means.labels_

    def get_missing_foothold_ids(self):
        return np.where(self.footholds == None)[0]
    
    def get_existing_foothold_ids(self):
        return np.where(self.footholds != None)[0]
    
    def get_existing_footholds(self):
        return self.footholds[np.where(self.footholds != None)[0]]
    
    def get_footholds_in_column(self, j):
        existing_hold_ids = self.get_existing_foothold_ids()
        existing_holds = self.get_existing_footholds()
        return existing_holds[existing_hold_ids%self.foot_box_length == j]

    def get_footholds_in_row(self, i):
        existing_hold_ids = self.get_existing_foothold_ids()
        existing_holds = self.get_existing_footholds()
        return existing_holds[np.floor(existing_hold_ids/self.foot_box_length) == i]
    
    def get_interpolated_missing_foothold(self, missing_hold_id):
        missing_hold_row = int(np.floor(missing_hold_id/self.foot_box_length))
        missing_hold_col = missing_hold_id%self.foot_box_length
        missing_hold_name = str(missing_hold_id)
        row = self.get_footholds_in_row(missing_hold_row)
        row_line_params = self.line_of_best_fit(row)
        col_line_params = self.odd_column_line_params[missing_hold_col]
        missing_hold_center_x = -(row_line_params[1]-col_line_params[1])/(row_line_params[0]-col_line_params[0])
        missing_hold_center_y= (row_line_params[1]*col_line_params[0]-row_line_params[0]*col_line_params[1])/(-row_line_params[0]+col_line_params[0])
        if np.isnan(missing_hold_center_x):
            raise ValueError('Slope or intercept in get_interpolated_missing_foothold() diverged when interpolating holds.')
        if np.isnan(missing_hold_center_y):
            raise ValueError('Slope or intercept in get_interpolated_missing_foothold() diverged when interpolating holds.')
        avg_hold_size = self.get_average_hold_size()
        missing_hold_top_left = (missing_hold_center_x - avg_hold_size[0]/2, missing_hold_center_y - avg_hold_size[1]/2)
        missing_hold_bottom_right = (missing_hold_center_x + avg_hold_size[0]/2, missing_hold_center_y + avg_hold_size[1]/2)
        missing_hold_region = (missing_hold_top_left[0],missing_hold_top_left[1],missing_hold_bottom_right[0] ,missing_hold_bottom_right[1])
        return Hold(missing_hold_name, missing_hold_region)
        
    def get_missing_handhold_ids(self):
        return np.where(self.handholds == None)[0]
    
    def get_missing_handholds(self):
        return self.handholds[np.where(self.handholds == None)[0]]
    
    def get_existing_handhold_ids(self):
        return np.where(self.handholds != None)[0]
    
    def get_existing_handholds(self):
        return self.handholds[np.where(self.handholds != None)[0]]
    
    def get_average_hold_size(self):
        existing_holds = self.get_existing_handholds()
        avg_hold_width = np.average(np.array(list(map(lambda x: x.region[2]-x.region[0], existing_holds))))
        avg_hold_height = np.average(np.array(list(map(lambda x: x.region[3]-x.region[1], existing_holds))))
        return (avg_hold_width,avg_hold_height)
    
    def get_handholds_in_column(self, j):
        existing_hold_ids = self.get_existing_handhold_ids()
        existing_holds = self.get_existing_handholds()
        return existing_holds[existing_hold_ids%self.hand_box_length == j]

    def get_handholds_in_row(self, i):
        existing_hold_ids = self.get_existing_handhold_ids()
        existing_holds = self.get_existing_handholds()
        return existing_holds[np.floor(existing_hold_ids/self.hand_box_length) == i]

    def get_interpolated_missing_handhold(self, missing_hold_id):
        missing_hold_row = int(np.floor(missing_hold_id/self.hand_box_length))
        missing_hold_col = missing_hold_id%self.hand_box_length
        missing_hold_name = chr(missing_hold_col+ ord('A'))+str(self.hand_box_width-missing_hold_row)
        col = self.get_handholds_in_column(missing_hold_col)
        row = self.get_handholds_in_row(missing_hold_row)
        row_line_params = self.line_of_best_fit(row)
        col_line_params = self.line_of_best_fit(col)
        missing_hold_center_x = -(row_line_params[1]-col_line_params[1])/(row_line_params[0]-col_line_params[0])
        missing_hold_center_y= (row_line_params[1]*col_line_params[0]-row_line_params[0]*col_line_params[1])/(-row_line_params[0]+col_line_params[0])
        if np.isnan(missing_hold_center_x):
            raise ValueError('Slope or intercept in get_interpolated_missing_handhold() diverged when interpolating holds.')
        if np.isnan(missing_hold_center_y):
            raise ValueError('Slope or intercept in get_interpolated_missing_handhold() diverged when interpolating holds.')
        avg_hold_size = self.get_average_hold_size()
        missing_hold_top_left = (missing_hold_center_x - avg_hold_size[0]/2, missing_hold_center_y - avg_hold_size[1]/2)
        missing_hold_bottom_right = (missing_hold_center_x + avg_hold_size[0]/2, missing_hold_center_y + avg_hold_size[1]/2)
        missing_hold_region = (missing_hold_top_left[0],missing_hold_top_left[1],missing_hold_bottom_right[0] ,missing_hold_bottom_right[1])
        return Hold(missing_hold_name, missing_hold_region)
    
    def line_of_best_fit(self, holds):
        hold_centers = np.array(list(map(lambda x: list(x.center), holds)))
        regression = LinearRegression().fit(hold_centers[:,0].reshape(-1,1),hold_centers[:,1])
        return [regression.coef_[0], regression.intercept_]

    @staticmethod
    def get_grid_index(name):
        shift = ord('a')
        col = (ord((name[0]).lower())-shift)
        row = 18 - int(name[1::])
        return (row,col)
    
    @staticmethod
    def get_hold_id(name):
        temp_grid_index = Board.get_grid_index(name)
        return temp_grid_index[0]*11 + temp_grid_index[1]

    @staticmethod
    def get_hold_name(id):
        shift = ord('A')
        row = str(18 - (int(np.floor(id/11))))
        col = chr(id%11 +shift)
        name = "" + col + row
        return name
    
    @staticmethod
    def holds_vector_representation(hold):
        temp = [0]*18*11
        if isinstance(hold,str):
            index = Board.get_hold_id(hold)
        temp[index] = 1
        return temp


class Hold:
    def __init__(self, name, region):
        shift = ord('A')
        if isinstance(name, int):
            row = str(18 - (int(np.floor(name/11))))
            col = chr(name%11 +shift)
            self.name = "" + col + row
        else:
            self.name = name
        self.region = region
        self.center = ((region[0]+region[2])/2,(region[1]+region[3])/2)

    def is_hand(self):
        if len(self.name) == 1:
            return False 
        else:
            return True   

    def get_grid_index(self):
        if self.is_hand():
            shift = ord('a')
            col = (ord((self.name[0]).lower())-shift)
            row = 18 - int(self.name[1::])
            return (row,col)
        return (-1,-1)

    def get_hold_id(self):
        temp_grid_index = self.get_grid_index()
        return temp_grid_index[0]*11 + temp_grid_index[1]
    
    def draw_hold_mask(self,draw,color):
        x1 = int(self.region[0])
        x2 = int(self.region[2])
        y1 = int(self.region[1])
        y2 = int(self.region[3])
        draw.rectangle((x1,y1,x2,y2),fill=color[0])
        draw.circle(self.center, fill=color[1], radius=2)

    def draw_hold(self, draw, color):
        x1 = int(self.region[0])
        x2 = int(self.region[2])
        y1 = int(self.region[1])
        y2 = int(self.region[3])
        draw.rectangle((x1,y1,x2,y2),outline=color[0])
        draw.circle(self.center, fill=color[1], radius=2)

    def distance_to_hold_line(self, line):
        p = self.center
        d = np.abs(-line[0]*p[0] + p[1] - line[1])/np.sqrt(line[0]**2 + 1)
        return d  
    
    def distance_to_hold(self, point):
        dx = max([self.region[0] - point[0], 0, point[0] - self.region[2]])
        dy = max([self.region[1] - point[1], 0, point[1] - self.region[3]])
        dx2 = self.center[0] - point[0]
        dy2 = self.center[1] - point[1]
        return np.sqrt(dx**2 + dy**2)+np.sqrt(dx2**2 + dy2**2)


class Person:
    def __init__(self, region, joints):
        self.region = region 
        self.joints = joints

    def get_center(self):
        person_region = self.region
        return np.array([person_region[0]+0.5*(person_region[2]-person_region[0]),person_region[1]+0.5*(person_region[3]-person_region[1])])

    def get_limbs(self):
        limbs = np.zeros((8,2))
        limbs[L_BICEP,:] = self.get_left_bicep()
        limbs[L_FOREARM,:] = self.get_left_forearm()
        limbs[R_BICEP,:] = self.get_right_bicep()
        limbs[R_FOREARM,:] = self.get_right_forearm()
        limbs[L_QUAD,:] = self.get_left_quad()
        limbs[L_CALF,:] = self.get_left_calf()
        limbs[R_QUAD,:] = self.get_right_quad()
        limbs[R_CALF,:] = self.get_right_calf()
        return limbs

    def outlier_joints(self):
        relevant_joints = self.joints[5:17,:]
        outliers = list(filter(lambda x: (x[0] < self.region[0] or x[0] > self.region[2]) or (x[1] < self.region[1] or x[1] > self.region[3]), relevant_joints))
        if len(outliers) > 0:
            return True 
        else:
            return False

    def get_left_forearm(self):
        left_elbow = self.joints[7,:]
        left_wrist = self.joints[9,:]
        return left_wrist - left_elbow

    def get_left_bicep(self):
        left_shoulder = self.joints[5,:]
        left_elbow = self.joints[7,:]
        return left_elbow - left_shoulder

    def get_right_forearm(self):
        right_elbow = self.joints[8,:]     
        right_wrist = self.joints[10,:]
        return right_wrist - right_elbow

    def get_right_bicep(self):
        right_shoulder = self.joints[6,:]
        right_elbow = self.joints[8,:]     
        return right_elbow - right_shoulder

    def get_left_quad(self):
        left_hip = self.joints[11,:]
        left_knee = self.joints[13,:]
        return left_knee - left_hip

    def get_left_calf(self):
        left_knee = self.joints[13,:]
        left_ankle = self.joints[15,:]
        return left_ankle - left_knee

    def get_right_quad(self):
        right_hip = self.joints[12,:]    
        right_knee = self.joints[14,:]
        return right_knee - right_hip

    def get_right_calf(self):  
        right_knee = self.joints[14,:]        
        right_ankle = self.joints[16,:]
        return right_ankle-right_knee