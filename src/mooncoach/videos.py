#from tkinter import *
#from tkinter import ttk

from ultralytics import YOLO
from roboflow import Roboflow
import h5py
import itertools as it
import copy
import torchvision as tv
from torchvision.transforms import v2

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from entities import *
from IPython import display 

class Video:
    def __init__(self):
        self.source = ""
        self.climb_in_database = False
        self.start_time = 0 
        self.frames = [] 
                
        self.valid_holds = []
        self.kickboard_in = False 
    
    def set_valid_holds(self,hold_keys):
        self.valid_holds = hold_keys
    
    def set_kickboard_in(self,kickboard_in):
        self.kickboard_in = kickboard_in

    def load_from_MP4(self, source, year, start_time, end_time):
        self.source = source
        path = "./../../datasets/climbing_videos/mp4/2024_shortuglybeta/"+source+".webm"
        self.start_time = start_time 
        print("Reading video.")
        video,_,_ = tv.io.read_video(path,start_pts = self.start_time,end_pts = end_time, pts_unit = 'sec',output_format='TCHW')
        print("Done Reading Video.")
        detect_holds = YOLO('./../../models/runs/detect/train13/weights/best.pt')
        detect_pose = YOLO('./../../models/yolov8m-pose.pt')
        transforms = v2.Compose([
            v2.Resize([640,640])
        ])
        images=transforms(video).float()
        frames = []
        count = 0
        print("Predicting pose and holds.")
        for i, x in enumerate(images):
            try:
                detect_holds_result = list(detect_holds(source=x.unsqueeze(0),stream=True, conf=0.7, boxes=False,verbose=False))[0]
                detect_pose_result = list(detect_pose(source=x.unsqueeze(0),stream=True,verbose=False))[0]
                detect_holds_result_image = detect_holds_result.plot(labels=False,boxes=False)
                hold_names = list(map(lambda x: detect_holds_result.names[x],detect_holds_result.boxes.cls.int().tolist()))
                hold_boxes = detect_holds_result.boxes.xyxy.detach().cpu().numpy().tolist()
                holds = np.array(list(it.starmap(Hold, list(zip(hold_names,hold_boxes)))))
                board = Board()
                board.set_holds_from_video(holds)
                #print(np.array(list(map(lambda x: x.region, board.footholds))))
                person = Person(detect_pose_result.boxes.xyxy.detach().cpu().numpy()[0], detect_pose_result.keypoints.xy.detach().cpu().numpy()[0])
                #frame = Frame(detect_holds_result_image, board, person)
                frame = Frame(board, person)
                self.frames.append(frame)
            except:
                print("Couldn't add frame {}.".format(i))
                count+=1
        print("Done predicting pose and holds.")
        if count > 0:
            print("Couldn't add {} frames.".format(count))
        
        """
        climb_name = source.split('⧸')[-1].split('.')[0]
        parsed = climb_name.split(':')[0].split(' ')
        if len(parsed) > 1:
            climb_name = ''.join(parsed[0:-1]).lower()
        else:
            climb_name = ''.join(parsed[0]).lower()
        """
        climb_name = ''.join(''.join(source.split('⧸')[0]).split(' ')[0:-1]).lower()

        hold_data = "./../../datasets/moonboard_sets/" + year + ".json"

        with open(hold_data) as f:
            d = json.load(f)
            climb_keys = list(d.keys())
            climb_names = list(map(lambda x: ''.join(d[x].get("Name").split(' ')).lower(), climb_keys))
            try:
                climb_index = climb_keys[climb_names.index(climb_name)] 
                self.kickboard_in = d[climb_index]['kickboardIn']
                
                valid_holds = list(map(lambda x: x, d[climb_index]['Moves']))
                valid_hold_names = list(map(lambda x: [x['Description'],x['IsStart'],x['IsEnd']], valid_holds))

                self.valid_holds = sorted(list(map(lambda x: [Board.get_hold_id(x[0]),x[1],x[2]], valid_hold_names)), key=lambda x:x[0])
                self.climb_in_database = True
            except: 
                print("Couldn't find " + climb_name + " in database.")
                self.climb_in_database = False

    def save_to_HDF5(self, output):
        with h5py.File(output, "a") as of:
            vid = of.create_group("video")
            dset0_1 = vid.create_dataset("kickboardIn", data=[self.kickboard_in])
            dset0_2 = vid.create_dataset("validHolds", data=np.array(self.valid_holds))
            dset0_2 = vid.create_dataset("inDatabase", data=[self.climb_in_database])
            for i,f in enumerate(self.frames):
                grp = of.create_group("frame_{}".format(i))
                grp1 = grp.create_group("person")
                dset1_1 = grp1.create_dataset("regions", data=np.array(f.person.region))
                dset1_2 = grp1.create_dataset("joints", data=np.array(f.person.joints))

                grp2 = grp.create_group("board")
                temp_footholds = np.array(list(map(lambda x: x.region, f.board.footholds)))
                dset2_1 = grp2.create_dataset("footHoldRegions", data=temp_footholds)
                temp_handholds= np.array(list(map(lambda x: x.region, f.board.handholds)))
                dset2_2 = grp2.create_dataset("handHoldRegions", data=temp_handholds)


    def load_from_HDF5(self, source):
        with h5py.File(source, "r") as of:
            foothold_ids = list(map(lambda x: str(x), range(10)))
            handhold_ids = list(range(11*18))
            frames = [None]*(len(of.keys())-1)
            for i,g in enumerate(of):
                if g != "video":
                    foothold_array = np.zeros((10,4))
                    of[g+"/board/footHoldRegions"].read_direct(foothold_array, np.s_[0:10,0:4], np.s_[0:10,0:4])
                    footholds = list(it.starmap(Hold,zip(foothold_ids,tuple(foothold_array.tolist()))))

                    handhold_array = np.zeros((11*18,4))
                    of[g+"/board/handHoldRegions"].read_direct(handhold_array, np.s_[0:11*18,0:4], np.s_[0:11*18,0:4])
                    handholds = list(it.starmap(Hold,zip(handhold_ids,tuple(handhold_array.tolist()))))
                    
                    person_region_array = np.zeros(4)
                    of[g+"/person/regions"].read_direct(person_region_array, np.s_[0:4], np.s_[0:4])
                    person_joint_array = np.zeros((17,2))
                    of[g+"/person/joints"].read_direct(person_joint_array, np.s_[0:17,0:2], np.s_[0:17,0:2])
                    person = Person(tuple(person_region_array.tolist()),person_joint_array)

                    new_board = Board()
                    list(map(new_board.set_foothold, footholds))
                    list(map(new_board.set_handhold, handholds))
                    frames[int(g.split('_')[1])] =Frame(new_board, person) 
                if g == "video":
                    self.climb_in_database = of["video/inDatabase"][0]
                    self.kickboard_in = of["video/kickboardIn"][0]
                    self.valid_holds = of["video/validHolds"][:]
            self.frames = frames
    
    def remove_outlier_frames(self):
        filtered_frames = list(filter(lambda x: x.person.outlier_joints() == False, self.frames))
        print("Original video had "+ str(len(self.frames)) +" frames.")
        print("Removed " + str((len(self.frames) - len(filtered_frames))) +" suspicious frames.")
        print("Current frame number: " + str(len(filtered_frames)))
        self.frames = filtered_frames

class ProcessedVideo:
    def __init__(self):
        self.source = ""
        self.board = Board() 
        self.frames = [] 
        self.new_fps_scale = 0

        self.moves = []

        self.valid_holds = []
        self.kickboard_in = False 
    
    def set_valid_holds(self,hold_keys):
        self.valid_holds = hold_keys
    
    def set_kickboard_in(self,kickboard_in):
        self.kickboard_in = kickboard_in

    def set_board(self, video):
        all_handholds = np.array(list(map(lambda x: x.board.handholds, video.frames)))
        all_footholds = np.array(list(map(lambda x: x.board.footholds, video.frames)))
        avg_handholds = np.apply_along_axis(self.average_holds, 0, all_handholds)
        list(map(self.board.set_handhold,avg_handholds))
        avg_footholds = np.apply_along_axis(self.average_holds, 0, all_footholds)
        list(map(self.board.set_foothold,avg_footholds))
        self.valid_holds = video.valid_holds
        self.kickboard_in = video.kickboard_in

    def set_frames(self, video, n):
        frames = np.array(list(map(lambda x: x.person, video.frames)))
        if n == 1:
            self.frames = frames
            self.new_fps_scale = 1
        else:
            num_old_frames = len(frames)
            new_frame_batch_size = n
            num_new_frames = int(np.floor(num_old_frames/new_frame_batch_size))
            self.new_fps_scale = num_old_frames/num_new_frames
            last_frame_batch_size = num_old_frames%n 
            if last_frame_batch_size == 0:
                temp = [new_frame_batch_size]*(num_new_frames-1)
                new_frame_structure = zip(range(num_new_frames-1),temp) 
            else:
                temp = [new_frame_batch_size]*(num_new_frames)
                temp[-1] = last_frame_batch_size
                new_frame_structure = zip(range(num_new_frames),temp)
            slice_set = list(map(lambda x: np.s_[x[0]*new_frame_batch_size:x[0]*new_frame_batch_size + x[1]], new_frame_structure))
            #print(slice_set)
            sliced_frames = list(map(lambda x: frames[x].tolist(), slice_set)) 
            self.frames = list(map(self.average_pose, sliced_frames))
        
        self.set_moves_from_time_series()
        final_hold = Board.get_hold_name(self.valid_holds[np.where(self.valid_holds[:,2] ==1)][0,0])
        if len(self.moves) > 0:
            final_hold_locations = np.where(np.array(list(map(lambda x: x.name, self.moves[:,2]))) == final_hold)[0]
            if len(final_hold_locations) > 0:
                final_move_frame = self.moves[np.max(final_hold_locations)][0]+1
                self.frames = self.frames[0:final_move_frame]

    def average_pose(self, people):
        n = len(people)
        regions = np.array(list(map(lambda x: x.region, people)))
        avg_regions = (1/n*np.sum(regions, axis=0)).tolist()
        joints = np.array(list(map(lambda x: x.joints, people)))
        avg_joints = 1/n*np.sum(joints, axis=0)
        return Person(avg_regions, avg_joints)

    def average_holds(self, holds):
        regions = np.array(list(map(lambda x: x.region, holds)))
        avg_regions = (1/len(holds)*np.sum(regions, axis=0)).tolist()
        avg_name = holds[0].name
        return Hold(avg_name, avg_regions)

    def play(self):
        fig, ax = plt.subplots(1,1, layout='compressed')
        valid_hold_ids = list(map(lambda x: x[0], self.valid_holds))
        #fig= plt.figure()
        ims = []

        draw_bg = Image.fromarray(np.zeros((640,640,3)),mode='RGB')
        draw = ImageDraw.Draw(draw_bg)
        fnt = ImageFont.truetype("./../../resources/arial.TTF", 20)

        n = 0
        for i,hold in enumerate(self.board.handholds):
            if i in valid_hold_ids:
                if (self.valid_holds[n][1] == True) and (self.valid_holds[n][2] == False): 
                    draw.rectangle(hold.region,fill=(0,250,0),width=4)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='lime')
                elif (self.valid_holds[n][1] == False) and (self.valid_holds[n][2] == True): 
                    draw.rectangle(hold.region,fill=(250,0,0),width=4)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='red')
                else:
                    draw.rectangle(hold.region,fill=(0,0,250),width=4)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='white')
                n+=1
            #else:
            #    draw.rectangle(hold.region,outline='grey',width=1)

        for hold in self.board.footholds:
            if self.kickboard_in == True:
                try:
                    draw.rectangle(hold.region,fill=(0,0,250),width=4)
                    #print(hold.region)
                    #print(hold.name)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='white')
                except:
                    print("couldnt draw")
            else:
                draw.rectangle(hold.region,outline='grey',width=1)

            
        for i,person in enumerate(self.frames):
            #print(person.joints)
            draw_on = copy.deepcopy(draw_bg)
            draw = ImageDraw.Draw(draw_on)

            person_region = person.region
            person_center = person.get_center()

            left_shoulder = person.joints[5,:]
            right_shoulder = person.joints[6,:]
            left_elbow = person.joints[7,:]
            right_elbow = person.joints[8,:]     
            left_wrist = person.joints[9,:]
            right_wrist = person.joints[10,:]  
            left_hip = person.joints[11,:]
            right_hip = person.joints[12,:]    
            left_knee = person.joints[13,:]
            right_knee = person.joints[14,:]        
            left_ankle = person.joints[15,:]
            right_ankle = person.joints[16,:] 

            draw.circle(left_shoulder, radius=4, fill=(0,250,250))
            draw.circle(right_shoulder, radius=4, fill=(250,0,250)) 
            
            draw.circle(left_hip, radius=4, fill=(250,0,250)) 
            draw.circle(right_hip, radius=4, fill=(250,250,0)) 
            
            draw.circle(left_elbow, radius=4, fill=(0,250,250)) 
            draw.circle(right_elbow, radius=4, fill=(250,0,250)) 
            
            draw.circle(left_wrist, radius=4, fill=(0,250,250)) 
            draw.circle(right_wrist, radius=4, fill=(250,0,250)) 
            
            draw.circle(left_ankle, radius=4, fill=(250,0,250))  
            draw.circle(left_knee, radius=4, fill=(250,0,250)) 


            draw.circle(right_ankle, radius=4, fill=(250,250,0))
            draw.circle(right_knee, radius=4, fill=(250,250,0))    

            #draw.circle(personCenter, radius=4, fill=None)  

            draw.line([tuple(left_hip),tuple(right_hip)], fill=(250,250,250), width=2)
            draw.line([tuple(left_hip),tuple(left_shoulder)], fill=(250,250,250), width=2)
            draw.line([tuple(left_shoulder),tuple(right_shoulder)], fill=(250,250,250), width=2)
            draw.line([tuple(right_shoulder),tuple(right_hip)], fill=(250,250,250), width=2)

            draw.line([tuple(right_shoulder),tuple(right_elbow)], fill=(250,0,250), width=2)
            draw.line([tuple(right_elbow),tuple(right_wrist)], fill=(250,0,250), width=2)

            draw.line([tuple(left_shoulder),tuple(left_elbow)], fill=(0,250,250), width=2)
            draw.line([tuple(left_elbow),tuple(left_wrist)], fill=(0,250,250), width=2)

            draw.line([tuple(left_hip),tuple(left_knee)], fill=(250,0,250), width=2)
            draw.line([tuple(left_knee),tuple(left_ankle)], fill=(250,0,250), width=2)

            draw.line([tuple(right_hip),tuple(right_knee)], fill=(250,250,0), width=2)
            draw.line([tuple(right_knee),tuple(right_ankle)], fill=(250,250,0), width=2)

            #draw.rectangle(personRegion, outline='white', width=2)

            im1 = plt.imshow(draw_on, animated=True)  # plot a BGR numpy array of predictions 
            #im1 = plt.imshow(draw_on, animated=True)  # plot a BGR numpy array of predictions 

            ims.append([im1])

        """
        fig2, ax2 = plt.subplots(2,1, layout='compressed')
        ims2 = []
        for i,person in enumerate(self.frames):
            im2 = ax2[0].axvline(x=i, color='red')
            im3 = ax2[1].axvline(x=i, color='red')
            ims2.append([im2,im3])

        ax2[0].plot(joint_time_series[0], label='Left Hand')
        ax2[0].plot(joint_time_series[1], label='Right Hand', linestyle='dashed')
        ax2[1].plot(joint_time_series[2], label='Left Foot')
        ax2[1].plot(joint_time_series[3], label='Right Foot',linestyle='dashed')
        """
        #ax2[2].plot(joint_time_series[4], label='Foot Movement')
        #ax2[2].plot(joint_time_series[5], label='Total',linestyle='dashed')
            
        #
        #ani = animation.ArtistAnimation(fig, ims, frames=62, interval = 50*self.new_fps_scale, blit = True, repeat_delay = 100)
        #ani2 = animation.ArtistAnimation(fig2, ims2, interval = 50*self.new_fps_scale, blit = True, repeat_delay = 100)
        ani = animation.ArtistAnimation(fig, ims, interval=300,repeat=False, blit=True)
        #ani2 = animation.ArtistAnimation(fig2, ims2, interval=10,repeat=True, blit = True)

        #ax2[0].legend()
        #ax2[1].legend()
        
        plt.show()
        #ani.save('everything6b.gif',writer=animation.PillowWriter(fps=10))
        #ani2.save('jointtimeseries.gif',writer=animation.PillowWriter(fps=10))
        #video = ani.to_html5_video()
        #video2 = ani.to_html5_video()
        #print(display.HTML(video))
        #print(display.HTML(video2))

    def joint_velocity_time_series(self):
        joint_time_series2 = np.gradient(np.array(list(map(lambda x: x.joints[5:17,:], self.frames))))[0]
        joint_time_series3 = np.gradient(joint_time_series2)[0]

        hand_kernel = np.ones(2)
        left_hand_vx = np.abs(-np.convolve(joint_time_series2[:,4][:,0],hand_kernel,mode='full')[0:len(joint_time_series2)])
        left_hand_vy = np.abs(-np.convolve(joint_time_series2[:,4][:,1],hand_kernel,mode='full')[0:len(joint_time_series2)])

        right_hand_vx = np.abs(-np.convolve(joint_time_series2[:,5][:,1],hand_kernel,mode='full')[0:len(joint_time_series2)])
        right_hand_vy = np.abs(-np.convolve(joint_time_series2[:,5][:,1],hand_kernel,mode='full')[0:len(joint_time_series2)])

        #left_hand_plot = left_hand_vx+ left_hand_vy
        #right_hand_plot = right_hand_vx+ right_hand_vy

        left_hand = np.where(left_hand_vx + left_hand_vy > 60, left_hand_vx + left_hand_vy, np.zeros(len(joint_time_series2)))
        right_hand = np.where(right_hand_vx + right_hand_vy > 60, right_hand_vx + right_hand_vy, np.zeros(len(joint_time_series2)))
        
        footMask = np.where(left_hand+right_hand > 60, np.zeros(len(joint_time_series2)), 1)

        
        foot_kernel = np.ones(2)
        left_foot_vx = np.abs(np.convolve(joint_time_series2[:,10][:,0],foot_kernel,mode='full')[0:len(joint_time_series2)]*footMask) #+ np.abs(np.convolve(joint_time_series3[:,10][:,0],foot_kernel,mode='full')[0:len(joint_time_series3)]*footMask)
        left_foot_vy = np.abs(np.convolve(joint_time_series2[:,10][:,1],foot_kernel,mode='full')[0:len(joint_time_series2)]*footMask) #+ np.abs(np.convolve(joint_time_series3[:,10][:,1],foot_kernel,mode='full')[0:len(joint_time_series3)]*footMask)
        left_foot = np.abs(left_foot_vx) + np.abs(left_foot_vy)

        right_foot_vx = np.abs(np.convolve(joint_time_series2[:,11][:,0],foot_kernel,mode='full')[0:len(joint_time_series2)]*footMask) #+ np.abs(np.convolve(joint_time_series3[:,11][:,0],foot_kernel,mode='full')[0:len(joint_time_series3)]*footMask)
        right_foot_vy = np.abs(np.convolve(joint_time_series2[:,11][:,1],foot_kernel,mode='full')[0:len(joint_time_series2)]*footMask) #+ np.abs(np.convolve(joint_time_series3[:,11][:,1],foot_kernel,mode='full')[0:len(joint_time_series3)]*footMask)
        right_foot = np.abs(right_foot_vx) + np.abs(right_foot_vy)

        left_hand = np.where(left_hand > 60, left_hand, np.zeros(len(joint_time_series2)))
        right_hand = np.where(right_hand > 60, right_hand, np.zeros(len(joint_time_series2)))

        left_hand_plot = np.ediff1d(np.where(left_hand > 50, np.ones(len(joint_time_series2)), np.zeros(len(joint_time_series2))),to_begin=[0])
        right_hand_plot = np.ediff1d(np.where(right_hand > 50, np.ones(len(joint_time_series2)), np.zeros(len(joint_time_series2))),to_begin=[0])
        left_hand_plot = np.where(left_hand_plot < 0, np.abs(left_hand_plot), np.zeros(len(joint_time_series3)))
        right_hand_plot = np.where(right_hand_plot < 0, np.abs(right_hand_plot), np.zeros(len(joint_time_series3)))

        """
        left_foot = np.where(left_foot > 30, left_foot, np.zeros(len(joint_time_series2)))
        right_foot = np.where(right_foot > 30, right_foot, np.zeros(len(joint_time_series2)))  
        
        filtered_left_foot = np.where(left_foot-right_foot > 0, np.abs(left_foot-right_foot)/np.max(np.abs(left_foot+right_foot))*left_foot, np.zeros(len(joint_time_series2)))
        filtered_right_foot = np.where(right_foot -left_foot > 0, np.abs(left_foot-right_foot)/np.max(np.abs(left_foot+right_foot))*right_foot, np.zeros(len(joint_time_series2)))  
        filtered_left_foot = left_foot 
        filtered_right_foot = right_foot

        foot_kernel = np.ones(5)
        filtered_left_foot = np.convolve(np.where(filtered_left_foot > 0, np.ones(len(joint_time_series3)), np.zeros(len(joint_time_series3))),foot_kernel,mode='full')[0:len(joint_time_series2)]
        filtered_right_foot = np.convolve(np.where(filtered_right_foot > 0, np.ones(len(joint_time_series3)), np.zeros(len(joint_time_series3))),foot_kernel,mode='full')[0:len(joint_time_series2)]
        temp_left_foot = np.where(filtered_left_foot > 0, np.ones(len(joint_time_series3)), np.zeros(len(joint_time_series2)))
        temp_right_foot = np.where(filtered_right_foot > 0, np.ones(len(joint_time_series3)), np.zeros(len(joint_time_series2)))
        #filtered_right_foot = filtered_left_foot
        filtered_right_foot_prime = np.where(np.gradient(temp_right_foot) < 0, np.abs(temp_right_foot), np.zeros(len(joint_time_series3)))
        filtered_left_foot_prime = np.where(np.gradient(temp_left_foot) < 0, np.abs(temp_left_foot), np.zeros(len(joint_time_series3)))

        foot_movement1 = filtered_left_foot 
        foot_movement2 = filtered_right_foot

        totalSignal = left_foot + right_foot + 0*(left_hand + right_hand)
        """
        return [left_hand_plot, right_hand_plot]#, filtered_left_foot_prime, filtered_right_foot_prime, foot_movement1, foot_movement2]
    
    def set_moves_from_time_series(self):
        #print(True)
        left_hand_moves = self.get_joint_moves("LH")
        right_hand_moves = self.get_joint_moves("RH")
        #left_foot_moves = self.get_joint_moves("LF")
        #right_foot_moves = self.get_joint_moves("RF")
        self.moves = np.array(sorted(left_hand_moves + right_hand_moves, key=lambda x: x[0]))
        #self.moves = np.array(sorted(left_hand_moves + right_hand_moves + left_foot_moves + right_foot_moves, key=lambda x: x[0]))
        #print(self.moves)

    def get_joint_moves(self, joint_name):
        frames = np.array(self.frames)
        joint_time_series = self.joint_velocity_time_series()
        moves = []
        if joint_name == "LH":
            key=np.where(joint_time_series[0]>0)
            joint_index = 9
        if joint_name == "RH":
            key=np.where(joint_time_series[1]>0)
            joint_index = 10
        if joint_name == "LF":
            key=np.where(joint_time_series[2]>0)
            joint_index = 15
        if joint_name == "RF":
            key=np.where(joint_time_series[3]>0)
            joint_index = 16
        nearest_hold = self.get_nearest_hold_to_hand(frames[0].joints[joint_index,:])
        moves.append([0, joint_name, nearest_hold])
        #print(moves)
        select_frames = frames[key]
        for i,f in enumerate(select_frames):
            n = len(moves)
            nearest_hold = self.get_nearest_hold_to_hand(f.joints[joint_index,:])   
            if moves[n-1][2].name != nearest_hold.name:
                moves.append([key[0].tolist()[i], joint_name, nearest_hold])
        return moves

    def get_nearest_hold_to_hand(self, joint_position):
        #print(joint_position)
        holds = np.concatenate((self.board.handholds, self.board.footholds))
        distances = list(map(lambda x: x.distance_to_hold(joint_position), holds))
        threshold = 8 # 4 nearest holds
        min_distance_holds = holds[np.argsort(distances)[:threshold]]
        #valid_min_distance_holds = np.where(Board.get_hold_id)
        min_distance_hold_names = list(map(lambda x: x.name, min_distance_holds))
        valid_holds = list(filter(self.check_if_hold_in_climb, min_distance_holds))
        if len(valid_holds) > 0:
            #print(list(map(lambda x: x.name, valid_holds)))
            return valid_holds[0]
        else:
            return min_distance_holds[0]

    def check_if_hold_in_climb(self, hold):
        if hold.is_hand():
            return hold.get_hold_id() in self.valid_holds 
        else:
            return self.kickboard_in
        
    def draw_moves(self):
        fig, ax = plt.subplots(1,1, layout='compressed')
        valid_hold_ids = list(map(lambda x: x[0], self.valid_holds))
        ims = []

        draw_bg = Image.fromarray(np.zeros((640,640,3)),mode='RGB')
        draw = ImageDraw.Draw(draw_bg)
        fnt = ImageFont.truetype("./../../resources/arial.TTF", 20)

        n = 0
        for i,hold in enumerate(self.board.handholds):
            if i in valid_hold_ids:
                if (self.valid_holds[n][1] == True) and (self.valid_holds[n][2] == False): 
                    draw.rectangle(hold.region,outline='lime',width=4)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='lime')
                elif (self.valid_holds[n][1] == False) and (self.valid_holds[n][2] == True): 
                    draw.rectangle(hold.region,outline='red',width=4)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='red')
                else:
                    draw.rectangle(hold.region,outline='white',width=4)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='white')
                n+=1
            else:
                draw.rectangle(hold.region,outline='grey',width=1)
        
        for hold in self.board.footholds:
            if self.kickboard_in == True:
                draw.rectangle(hold.region,outline='white',width=4)
                draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='white')
            else:
                draw.rectangle(hold.region,outline='grey',width=1)

        draw_on = copy.deepcopy(draw_bg)
        draw = ImageDraw.Draw(draw_on)
        LH = self.moves[0,2].center
        RH = self.moves[1,2].center
        LF = self.moves[2,2].center
        RF = self.moves[3,2].center
        im1 = plt.imshow(draw_on, animated=True)  # plot a BGR numpy array of predictions 

        ims.append([im1])
        
        for i in range(4,len(self.moves)):
            draw_on = copy.deepcopy(draw_bg)
            draw = ImageDraw.Draw(draw_on)
            
            joint_name = self.moves[i][1]
            if joint_name == "LH":
                LH = self.moves[i,2].center
            if joint_name == "RH":
                RH = self.moves[i,2].center
            if joint_name == "LF":
                LF = self.moves[i,2].center
            if joint_name == "RF":
                RF = self.moves[i,2].center

            if LF == RF:
                LF = (LF[0]-15, LF[1])
                RF = (LF[0]+15, LF[1])
            if LH == RH:
                LH = (LH[0]-15, LH[1])
                RH = (LH[0]+15, LH[1])

            mid_shoulder = np.array([(RH[0]+LH[0])/2, (RH[1]+LH[1])/2])
            mid_hip = np.array([(RF[0]+LF[0])/2, (RF[1]+LF[1])/2])

            torso = mid_shoulder - mid_hip 
            neck = mid_shoulder - 0.25*torso
            neck = (neck[0],neck[1])
            waist = mid_hip + 0.1*torso
            waist = (waist[0],waist[1])

            #head = mid_shoulder + 0.1*torso
            #head = (head[0], head[1])

            draw.circle(LH, radius=4, fill='red')
            draw.circle(RH, radius=4, fill='red') 
            draw.circle(LF, radius=4, fill='red') 
            draw.circle(RF, radius=4, fill='red') 
            draw.line([LH,neck], fill='lime', width=2)
            draw.line([RH,neck], fill='lime', width=2)
            draw.line([LF,waist], fill='lime', width=2)
            draw.line([RF,waist], fill='lime', width=2)
            draw.line([waist,neck], fill='lime', width=2)
            #draw.circle(head, radius=10, outline='lime') 
            #draw.rectangle(personRegion, outline='white', width=2)

            im1 = plt.imshow(draw_on, animated=True)  # plot a BGR numpy array of predictions 
            #im1 = plt.imshow(draw_on, animated=True)  # plot a BGR numpy array of predictions 

            ims.append([im1])
        ani = animation.ArtistAnimation(fig, ims, interval=500,repeat=True, blit=True)
        plt.show()

    def draw_moves_accurate(self):
        fig, ax = plt.subplots(1,1, layout='compressed')
        valid_hold_ids = list(map(lambda x: x[0], self.valid_holds))
        #fig= plt.figure()
        ims = []

        draw_bg = Image.fromarray(np.zeros((640,640,3)),mode='RGB')
        draw = ImageDraw.Draw(draw_bg)
        fnt = ImageFont.truetype("./../../resources/arial.TTF", 20)

        n = 0
        for i,hold in enumerate(self.board.handholds):
            if i in valid_hold_ids:
                if (self.valid_holds[n][1] == True) and (self.valid_holds[n][2] == False): 
                    draw.rectangle(hold.region,outline='lime',width=4)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='lime')
                elif (self.valid_holds[n][1] == False) and (self.valid_holds[n][2] == True): 
                    draw.rectangle(hold.region,outline='red',width=4)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='red')
                else:
                    draw.rectangle(hold.region,outline='white',width=4)
                    draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='white')
                n+=1
            else:
                draw.rectangle(hold.region,outline='grey',width=1)
        
        for hold in self.board.footholds:
            if self.kickboard_in == True:
                draw.rectangle(hold.region,outline='white',width=4)
                draw.text((hold.region[2],hold.region[3]), hold.name, font=fnt, fill='white')
            else:
                draw.rectangle(hold.region,outline='grey',width=1)

        frames = np.array(self.frames)[self.moves[:,0].astype(int)]
        
        for i,person in enumerate(frames):
            draw_on = copy.deepcopy(draw_bg)
            draw = ImageDraw.Draw(draw_on)

            person_region = person.region
            person_center = person.get_center()

            left_shoulder = person.joints[5,:]
            right_shoulder = person.joints[6,:]
            left_elbow = person.joints[7,:]
            right_elbow = person.joints[8,:]     
            left_wrist = person.joints[9,:]
            right_wrist = person.joints[10,:]  
            left_hip = person.joints[11,:]
            right_hip = person.joints[12,:]    
            left_knee = person.joints[13,:]
            right_knee = person.joints[14,:]        
            left_ankle = person.joints[15,:]
            right_ankle = person.joints[16,:] 

            draw.circle(left_shoulder, radius=4, fill='lime')
            draw.circle(right_shoulder, radius=4, fill='lime') 
            draw.circle(left_hip, radius=4, fill='lime') 
            draw.circle(right_hip, radius=4, fill='lime') 
            draw.circle(left_elbow, radius=4, fill='yellow') 
            draw.circle(right_elbow, radius=4, fill='yellow') 
            draw.circle(left_wrist, radius=4, fill='red') 
            draw.circle(right_wrist, radius=4, fill='red') 
            draw.circle(left_ankle, radius=4, fill='red') 
            draw.circle(right_ankle, radius=4, fill='red') 
            draw.circle(left_knee, radius=4, fill='yellow') 
            draw.circle(right_knee, radius=4, fill='yellow')    

            #draw.circle(personCenter, radius=4, fill=None)  

            draw.line([tuple(left_hip),tuple(right_hip)], fill='lime', width=2)
            draw.line([tuple(left_hip),tuple(left_shoulder)], fill='lime', width=2)
            draw.line([tuple(left_shoulder),tuple(right_shoulder)], fill='lime', width=2)
            draw.line([tuple(right_shoulder),tuple(right_hip)], fill='lime', width=2)

            draw.line([tuple(right_shoulder),tuple(right_elbow)], fill='yellow', width=2)
            draw.line([tuple(left_shoulder),tuple(left_elbow)], fill='yellow', width=2)
            draw.line([tuple(right_elbow),tuple(right_wrist)], fill='yellow', width=2)
            draw.line([tuple(left_elbow),tuple(left_wrist)], fill='yellow', width=2)

            draw.line([tuple(left_hip),tuple(left_knee)], fill='yellow', width=2)
            draw.line([tuple(right_hip),tuple(right_knee)], fill='yellow', width=2)
            draw.line([tuple(left_knee),tuple(left_ankle)], fill='yellow', width=2)
            draw.line([tuple(right_knee),tuple(right_ankle)], fill='yellow', width=2)

            #draw.rectangle(personRegion, outline='white', width=2)
            draw.text((30,30),str(self.moves[:,0][i]), font=fnt, fill='white')

            im1 = plt.imshow(draw_on, animated=True)  # plot a BGR numpy array of predictions 
            #im1 = plt.imshow(draw_on, animated=True)  # plot a BGR numpy array of predictions 

            ims.append([im1])

        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat=True, blit=True)

        plt.show()