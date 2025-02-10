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

class Video:
    def __init__(self):
        self.source = ""
        self.startTime = 0 
        self.endTime = 0 
        self.frames = [] 

    def loadFromMP4(self, source, startTime, endTime):
        self.source = source
        self.startTime = startTime 
        self.endTime = endTime
        print("Reading video.")
        video,_,_ = tv.io.read_video(self.source,start_pts = self.startTime, pts_unit = 'sec',output_format='TCHW')
        print("Done Reading Video.")
        detectHolds = YOLO('./../../models/runs/detect/train13/weights/best.pt')
        detectPose = YOLO('./../../models/yolov8m-pose.pt')
        transforms = v2.Compose([
            v2.Resize([640,640])
        ])
        images=transforms(video).float()
        frames = []
        count = 0
        print("Predicting pose and holds.")
        for i, x in enumerate(images):
            try:
                detectHoldsResult = list(detectHolds(source=x.unsqueeze(0),stream=True, conf=0.7, boxes=False))[0]
                detectPoseResult = list(detectPose(source=x.unsqueeze(0),stream=True))[0]
                detectHoldsResultImage = detectHoldsResult.plot(labels=False,boxes=False)
                holdNames = list(map(lambda x: detectHoldsResult.names[x],detectHoldsResult.boxes.cls.int().tolist()))
                holdBoxes = detectHoldsResult.boxes.xyxy.detach().cpu().numpy().tolist()
                holds = np.array(list(it.starmap(Hold, list(zip(holdNames,holdBoxes)))))
                board = Board()
                board.setHoldsFromVideo(holds)  
                person = Person(detectPoseResult.boxes.xyxy.detach().cpu().numpy()[0], detectPoseResult.keypoints.xy.detach().cpu().numpy()[0])
                #frame = Frame(detectHoldsResultImage, board, person)
                frame = Frame(board, person)
                self.frames.append(frame)
            except:
                print("Couldn't add frame {}.".format(i))
                count+=1
        print("Done predicting pose and holds.")
        if count > 0:
            print("Couldn't add {} frames.".format(count))


    def play(self):
        fig = plt.figure()
        ims = []
        for frame in self.frames:
            #drawOn = Image.fromarray(frame.img)
            drawOn = Image.fromarray(np.zeros((640,640)))
            draw = ImageDraw.Draw(drawOn)
            #fnt = ImageFont.truetype("/Library/Fonts/arial.TTF", 40)

            for hold in frame.board.getExistingHandHolds():
                draw.rectangle(hold.region,outline='blue',width=4)
                #draw.text(hold.center, hold.name, font=fnt, fill='white')
                
            for hold in frame.board.getExistingFootHolds():
                draw.rectangle(hold.region,outline='blue',width=4)
                #draw.text(hold.center, hold.name, font=fnt, fill='white')

            personRegion = frame.person.region
            personCenter = frame.person.getCenter()

            leftShoulder = frame.person.joints[5,:]
            rightShoulder = frame.person.joints[6,:]
            leftElbow = frame.person.joints[7,:]
            rightElbow = frame.person.joints[8,:]     
            leftWrist = frame.person.joints[9,:]
            rightWrist = frame.person.joints[10,:]  
            leftHip = frame.person.joints[11,:]
            rightHip = frame.person.joints[12,:]    
            leftKnee = frame.person.joints[13,:]
            rightKnee = frame.person.joints[14,:]        
            leftAnkle = frame.person.joints[15,:]
            rightAnkle = frame.person.joints[16,:] 

            draw.circle(leftShoulder, radius=4, fill='lime')
            draw.circle(rightShoulder, radius=4, fill='lime') 
            draw.circle(leftHip, radius=4, fill='lime') 
            draw.circle(rightHip, radius=4, fill='lime') 
            draw.circle(leftElbow, radius=4, fill='yellow') 
            draw.circle(rightElbow, radius=4, fill='yellow') 
            draw.circle(leftWrist, radius=4, fill='red') 
            draw.circle(rightWrist, radius=4, fill='red') 
            draw.circle(leftAnkle, radius=4, fill='red') 
            draw.circle(rightAnkle, radius=4, fill='red') 
            draw.circle(leftKnee, radius=4, fill='yellow') 
            draw.circle(rightKnee, radius=4, fill='yellow')    

            draw.circle(personCenter, radius=4, fill=None)  

            draw.line([tuple(leftHip),tuple(rightHip)], fill='lime', width=2)
            draw.line([tuple(leftHip),tuple(leftShoulder)], fill='lime', width=2)
            draw.line([tuple(leftShoulder),tuple(rightShoulder)], fill='lime', width=2)
            draw.line([tuple(rightShoulder),tuple(rightHip)], fill='lime', width=2)

            draw.line([tuple(rightShoulder),tuple(rightElbow)], fill='yellow', width=2)
            draw.line([tuple(leftShoulder),tuple(leftElbow)], fill='yellow', width=2)            
            leftShoulder = frame.person.joints[5,:]
            rightShoulder = frame.person.joints[6,:]
            leftElbow = frame.person.joints[7,:]
            rightElbow = frame.person.joints[8,:]     
            leftWrist = frame.person.joints[9,:]
            rightWrist = frame.person.joints[10,:]  
            leftHip = frame.person.joints[11,:]
            rightHip = frame.person.joints[12,:]    
            leftKnee = frame.person.joints[13,:]
            rightKnee = frame.person.joints[14,:]        
            leftAnkle = frame.person.joints[15,:]
            rightAnkle = frame.person.joints[16,:] 

    def saveToHDF5(self, output):
        with h5py.File(output, "a") as of:
            for i,f in enumerate(self.frames):
                grp = of.create_group("frame_{}".format(i))
                grp1 = grp.create_group("person")
                dset1_1 = grp1.create_dataset("regions", data=np.array(f.person.region))
                dset1_2 = grp1.create_dataset("joints", data=np.array(f.person.joints))

                grp2 = grp.create_group("board")
                tempFootHolds = np.array(list(map(lambda x: x.region, f.board.footHolds)))
                dset2_1 = grp2.create_dataset("footHoldRegions", data=tempFootHolds)
                tempHandHolds= np.array(list(map(lambda x: x.region, f.board.handHolds)))
                dset2_2 = grp2.create_dataset("handHoldRegions", data=tempHandHolds)

    def loadFromHDF5(self, source):
        with h5py.File(source, "r") as of:
            footHoldIds = list(map(lambda x: str(x), range(10)))
            handHoldIds = list(range(11*18))
            frames = [None]*len(of.keys())
            for g in of:
                footHoldArray = np.zeros((10,4))
                of[g]["board/footHoldRegions"].read_direct(footHoldArray, np.s_[0:10,0:4], np.s_[0:10,0:4])
                footHolds = list(it.starmap(Hold,zip(footHoldIds,tuple(footHoldArray.tolist()))))

                handHoldArray = np.zeros((11*18,4))
                of[g]["board/handHoldRegions"].read_direct(handHoldArray, np.s_[0:11*18,0:4], np.s_[0:11*18,0:4])
                handHolds = list(it.starmap(Hold,zip(handHoldIds,tuple(handHoldArray.tolist()))))
                
                personRegionArray = np.zeros(4)
                of[g]["person/regions"].read_direct(personRegionArray, np.s_[0:4], np.s_[0:4])
                personJointArray = np.zeros((17,2))
                of[g]["person/joints"].read_direct(personJointArray, np.s_[0:17,0:2], np.s_[0:17,0:2])
                person = Person(tuple(personRegionArray.tolist()),personJointArray)

                newBoard = Board()
                list(map(newBoard.setFootHold, footHolds))
                list(map(newBoard.setHandHold, handHolds))
                frames[int(g.split('_')[1])] =Frame(newBoard, person) 
            self.frames = frames
    
    def removeOutlierFrames(self):
        filteredFrames = list(filter(lambda x: x.person.outlierJoints() == False, self.frames))
        print("Original video had "+ str(len(self.frames)) +" frames.")
        print("Removed " + str((len(self.frames) - len(filteredFrames))) +" suspicious frames.")
        print("Current frame number: " + str(len(filteredFrames)))
        self.frames = filteredFrames

class Pvideo:
    def __init__(self):
        self.source = ""
        self.startTime = 0 
        self.endTime = 0
        self.board = Board() 
        self.frames = [] 
        self.newFpsScale = 0

        self.handBoxWidth = 18 
        self.handBoxLength = 11
        self.footBoxWidth = 2 
        self.footBoxLength = 5
        self.handBoxDimension = self.handBoxWidth*self.handBoxLength
        self.footBoxDimension = self.footBoxWidth*self.footBoxLength
        self.handHolds = np.empty(self.handBoxDimension,dtype=object)
        self.footHolds = np.empty(self.footBoxDimension,dtype=object)

    def setBoard(self, video):
        allHandHolds = np.array(list(map(lambda x: x.board.handHolds, video.frames)))
        allFootHolds = np.array(list(map(lambda x: x.board.footHolds, video.frames)))
        avgHandHolds = np.apply_along_axis(self.averageHolds, 0, allHandHolds)
        list(map(self.board.setHandHold,avgHandHolds))
        avgFootHolds = np.apply_along_axis(self.averageHolds, 0, allFootHolds)
        list(map(self.board.setFootHold,avgFootHolds))

    def setFrames(self, video, n):
        frames = np.array(list(map(lambda x: x.person, video.frames)))
        if n == 1:
            self.frames = frames
            self.newFpsScale = 1
        else:
            numOldFrames = len(frames)
            newFrameBatchSize = n
            numNewFrames = int(np.floor(numOldFrames/newFrameBatchSize))
            self.newFpsScale = numOldFrames/numNewFrames
            lastFrameBatchSize = numOldFrames%n 
            if lastFrameBatchSize == 0:
                temp = [newFrameBatchSize]*(numNewFrames-1)
                newFrameStructure = zip(range(numNewFrames-1),temp) 
            else:
                temp = [newFrameBatchSize]*(numNewFrames)
                temp[-1] = lastFrameBatchSize
                newFrameStructure = zip(range(numNewFrames),temp) 
            sliceSet = list(map(lambda x: np.s_[x[0]*newFrameBatchSize:x[0]*newFrameBatchSize + x[1]], newFrameStructure))
            slicedFrames = list(map(lambda x: frames[x].tolist(), sliceSet)) 
            self.frames = list(map(self.averagePose, slicedFrames))

    def averagePose(self, people):
        n = len(people)
        regions = np.array(list(map(lambda x: x.region, people)))
        avgRegions = (1/n*np.sum(regions, axis=0)).tolist()
        joints = np.array(list(map(lambda x: x.joints, people)))
        avgJoints = 1/n*np.sum(joints, axis=0)
        return Person(avgRegions, avgJoints)

    def averageHolds(self, holds):
        regions = np.array(list(map(lambda x: x.region, holds)))
        avgRegions = (1/len(holds)*np.sum(regions, axis=0)).tolist()
        avgName = holds[0].name
        return Hold(avgName, avgRegions)

    def play(self):
        fig, ax = plt.subplots(1,1, layout='compressed')
        
        
        #fig= plt.figure()
        ims = []

        drawBg = Image.fromarray(np.zeros((640,640)))
        draw = ImageDraw.Draw(drawBg)
        #fnt = ImageFont.truetype("/Library/Fonts/arial.TTF", 40)

        for hold in self.board.getExistingHandHolds():
            draw.rectangle(hold.region,outline='blue',width=4)
            #draw.text(hold.center, hold.name, font=fnt, fill='white')
        
        for hold in self.board.getExistingFootHolds():
            draw.rectangle(hold.region,outline='blue',width=4)
            #draw.text(hold.center, hold.name, font=fnt, fill='white')
        
        jointTimeSeries = self.jointVelocityTimeSeries()
            
        for i,person in enumerate(self.frames):
            drawOn = copy.deepcopy(drawBg)
            draw = ImageDraw.Draw(drawOn)

            personRegion = person.region
            personCenter = person.getCenter()

            leftShoulder = person.joints[5,:]
            rightShoulder = person.joints[6,:]
            leftElbow = person.joints[7,:]
            rightElbow = person.joints[8,:]     
            leftWrist = person.joints[9,:]
            rightWrist = person.joints[10,:]  
            leftHip = person.joints[11,:]
            rightHip = person.joints[12,:]    
            leftKnee = person.joints[13,:]
            rightKnee = person.joints[14,:]        
            leftAnkle = person.joints[15,:]
            rightAnkle = person.joints[16,:] 

            draw.circle(leftShoulder, radius=4, fill='lime')
            draw.circle(rightShoulder, radius=4, fill='lime') 
            draw.circle(leftHip, radius=4, fill='lime') 
            draw.circle(rightHip, radius=4, fill='lime') 
            draw.circle(leftElbow, radius=4, fill='yellow') 
            draw.circle(rightElbow, radius=4, fill='yellow') 
            draw.circle(leftWrist, radius=4, fill='red') 
            draw.circle(rightWrist, radius=4, fill='red') 
            draw.circle(leftAnkle, radius=4, fill='red') 
            draw.circle(rightAnkle, radius=4, fill='red') 
            draw.circle(leftKnee, radius=4, fill='yellow') 
            draw.circle(rightKnee, radius=4, fill='yellow')    

            draw.circle(personCenter, radius=4, fill=None)  

            draw.line([tuple(leftHip),tuple(rightHip)], fill='lime', width=2)
            draw.line([tuple(leftHip),tuple(leftShoulder)], fill='lime', width=2)
            draw.line([tuple(leftShoulder),tuple(rightShoulder)], fill='lime', width=2)
            draw.line([tuple(rightShoulder),tuple(rightHip)], fill='lime', width=2)

            draw.line([tuple(rightShoulder),tuple(rightElbow)], fill='yellow', width=2)
            draw.line([tuple(leftShoulder),tuple(leftElbow)], fill='yellow', width=2)
            draw.line([tuple(rightElbow),tuple(rightWrist)], fill='yellow', width=2)
            draw.line([tuple(leftElbow),tuple(leftWrist)], fill='yellow', width=2)

            draw.line([tuple(leftHip),tuple(leftKnee)], fill='orange', width=2)
            draw.line([tuple(rightHip),tuple(rightKnee)], fill='orange', width=2)
            draw.line([tuple(leftKnee),tuple(leftAnkle)], fill='orange', width=2)
            draw.line([tuple(rightKnee),tuple(rightAnkle)], fill='orange', width=2)

            draw.rectangle(personRegion, outline='white', width=2)

            im1 = plt.imshow(drawOn, animated=True)  # plot a BGR numpy array of predictions 
            im1 = plt.imshow(drawOn, animated=True)  # plot a BGR numpy array of predictions 

            ims.append([im1])


        fig2, ax2 = plt.subplots(2,1, layout='compressed')
        ims2 = []
        for i,person in enumerate(self.frames):
            if i == len(self.frames)-1:
                ax2[0].plot(jointTimeSeries[0], label='Left Hand')
                ax2[0].plot(jointTimeSeries[1], label='Right Hand')
                ax2[1].plot(jointTimeSeries[2], label='Left Foot')
                ax2[1].plot(jointTimeSeries[3], label='Right Foot')
            else:
                ax2[0].plot(jointTimeSeries[0])
                ax2[0].plot(jointTimeSeries[1])
                ax2[1].plot(jointTimeSeries[2])
                ax2[1].plot(jointTimeSeries[3])
            
            im2 = ax2[0].axvline(x=i, color='red')
            im3 = ax2[1].axvline(x=i, color='red')
            ims2.append([im2,im3])

        ani = animation.ArtistAnimation(fig, ims, interval = 50*self.newFpsScale, blit = True, repeat_delay = 100)
        ani2 = animation.ArtistAnimation(fig2, ims2, interval = 50*self.newFpsScale, blit = True, repeat_delay = 100)
        #mywriter = animation.FFMpegWriter(fps=30)
        #ani.save('myanimation.mp4',writer=mywriter)
        ax2[0].legend()
        ax2[1].legend()
        plt.show()

    def jointVelocityTimeSeries(self):
        jointTimeSeries2 = np.gradient(np.array(list(map(lambda x: x.joints[5:17,:], self.frames))))[0]
        #jointTimeSeries2 = np.array(list(map(lambda x: x.joints[5:17,:], self.frames)))
        x = np.linspace(0,len(jointTimeSeries2),len(jointTimeSeries2))
        """
        kernal = gaussian(x,0,50)
        kernal2 = np.ones(75)
        leftHandVx = np.convolve(-np.convolve(jointTimeSeries2[:,4][:,0],kernal,mode='full')[0:len(jointTimeSeries2)],kernal2,mode='full')[0:len(jointTimeSeries2)]
        leftHandVy = np.convolve(-np.convolve(jointTimeSeries2[:,4][:,1],kernal,mode='full')[0:len(jointTimeSeries2)],kernal2,mode='full')[0:len(jointTimeSeries2)]

        rightHandVx = np.convolve(-np.convolve(jointTimeSeries2[:,5][:,1],kernal,mode='full')[0:len(jointTimeSeries2)],kernal2,mode='full')[0:len(jointTimeSeries2)]
        rightHandVy = np.convolve(-np.convolve(jointTimeSeries2[:,5][:,1],kernal,mode='full')[0:len(jointTimeSeries2)],kernal2,mode='full')[0:len(jointTimeSeries2)]
        leftFootVx = np.convolve(-np.convolve(jointTimeSeries2[:,10][:,0],kernal,mode='full')[0:len(jointTimeSeries2)],kernal2,mode='full')[0:len(jointTimeSeries2)]
        """

        kernel = np.ones(3)
        leftHandVx = np.abs(-np.convolve(jointTimeSeries2[:,4][:,0],kernel,mode='full')[0:len(jointTimeSeries2)])
        leftHandVy = np.abs(-np.convolve(jointTimeSeries2[:,4][:,1],kernel,mode='full')[0:len(jointTimeSeries2)])

        rightHandVx = np.abs(-np.convolve(jointTimeSeries2[:,5][:,1],kernel,mode='full')[0:len(jointTimeSeries2)])
        rightHandVy = np.abs(-np.convolve(jointTimeSeries2[:,5][:,1],kernel,mode='full')[0:len(jointTimeSeries2)])

        leftHand = np.where(leftHandVx + leftHandVy > 50, leftHandVx + leftHandVy, np.zeros(len(jointTimeSeries2)))
        rightHand = np.where(rightHandVx + rightHandVy > 50, rightHandVx + rightHandVy, np.zeros(len(jointTimeSeries2)))
        
        footMask = np.where(leftHand+rightHand > 50, np.zeros(len(jointTimeSeries2)), 1)

        #kernel = np.ones(10)
        ##leftFootVx = np.convolve(jointTimeSeries2[:,10][:,0],kernel,mode='full')[0:len(jointTimeSeries2)]*footMask
        #leftFootVy = np.convolve(jointTimeSeries2[:,11][:,1],kernel,mode='full')[0:len(jointTimeSeries2)]*footMask
        
        kernel = np.ones(10)
        leftFootVx = np.convolve(jointTimeSeries2[:,10][:,0]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]*footMask
        leftFootVy = np.convolve(jointTimeSeries2[:,10][:,1]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]*footMask

        #leftFootVx = np.convolve(np.convolve(jointTimeSeries2[:,10][:,0]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]
        #leftFootVy = np.convolve(np.convolve(jointTimeSeries2[:,10][:,1]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]
        
        rightFootVx = np.convolve(jointTimeSeries2[:,11][:,0]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]*footMask
        rightFootVy = np.convolve(jointTimeSeries2[:,11][:,1]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]*footMask
        #rightFootVx = np.convolve(np.convolve(jointTimeSeries2[:,11][:,0]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]
        #rightFootVy = np.convolve(np.convolve(jointTimeSeries2[:,11][:,1]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]*footMask,kernel,mode='full')[0:len(jointTimeSeries2)]
        
        jointTimeSeries = np.linalg.norm(np.gradient(np.gradient(np.array(list(map(lambda x: x.joints[5:17,:], self.frames))))[0],axis=2),axis=2)
        avgJointVelocity = np.sum(jointTimeSeries,axis=1)/12
        avgHandVelocity = (jointTimeSeries[:,4]+jointTimeSeries[:,5])/2
        #avgHandVelocity = (jointTimeSeries[:,0]*jointTimeSeries[:,2]*jointTimeSeries[:,4]+jointTimeSeries[:,1]*jointTimeSeries[:,3]*jointTimeSeries[:,5])/2
        avgFootVelocity = (jointTimeSeries[:,10]+jointTimeSeries[:,11])/2
        #print(len(avgJointVelocity))
        centerTimeSeries = np.linalg.norm(np.gradient(np.array(list(map(lambda x: x.getCenter(), self.frames))))[0],axis=1)
        return [leftHand, rightHand, leftFootVx+ leftFootVy, rightFootVx+rightFootVy]