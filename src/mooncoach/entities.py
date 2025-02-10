import numpy as np
from sklearn.cluster import KMeans 
from sklearn.linear_model import LinearRegression
from PIL import Image, ImageDraw, ImageFont

LBICEP = 0
LFOREARM = 1
RBICEP = 2
RFOREARM = 3
LQUAD = 4
LCALF = 5
RQUAD = 6 
RCALF = 7

class Frame:
    def __init__(self, board, person):
        #self.img = img 
        self.board = board 
        self.person = person

class Board:
    def __init__(self):
        self.handBoxWidth = 18 
        self.handBoxLength = 11
        self.footBoxWidth = 2 
        self.footBoxLength = 5
        self.handBoxDimension = self.handBoxWidth*self.handBoxLength
        self.footBoxDimension = self.footBoxWidth*self.footBoxLength
        self.handHolds = np.empty(self.handBoxDimension,dtype=object)
        self.footHolds = np.empty(self.footBoxDimension,dtype=object)
        self.boardMesh = np.zeros((640,640,3))
        #self.im = Image.fromarray(self.boardMesh, mode='RGB')
        #self.draw = ImageDraw.Draw(self.im)
        self.oddColumn = []
        self.oddColumnLineParams = []

    def setHoldsFromVideo(self,holds):
        list(map(self.setHandHold,holds))
        missingHolds = list(map(self.getInterpolatedMissingHandHold, self.getMissingHandHoldIds()))
        list(map(self.setHandHold,missingHolds))

        self.setFootHoldLines()
        self.detectFootHolds(holds)
        missingFootHolds = list(map(self.getInterpolatedMissingFootHold, self.getMissingFootHoldIds()))
        list(map(self.setFootHold,missingFootHolds))

    def setHandHold(self,hold):
        if hold.isHand():
            holdIndex = hold.getHoldId()
            self.handHolds[holdIndex] = hold

    def setFootHold(self,hold):
        if hold.isHand() == False:
            holdIndex = int(hold.name)
            self.footHolds[holdIndex] = hold

    def setFootHoldLines(self):
        self.oddColumn = list(map(self.getHandHoldsInColumn,[1,3,5,7,9]))
        self.oddColumnLineParams = list(map(self.lineOfBestFit, self.oddColumn)) 

    def detectFootHolds(self, holds):
        footHolds = np.array(list(filter(lambda x: x.isHand() == False, holds)))
        bottomRow = footHolds[self.clusterHoldsY(footHolds) == 1]
        topRow = footHolds[self.clusterHoldsY(footHolds) == 0]
        for hold in bottomRow:
            holdToLineDistances = np.array(list(map(hold.distanceToHoldLine, self.oddColumnLineParams)))
            holdColumn = holdToLineDistances.argsort()[0]
            self.footHolds[holdColumn + 5] = hold
        for hold in topRow:
            holdToLineDistances = np.array(list(map(hold.distanceToHoldLine, self.oddColumnLineParams)))
            holdColumn = holdToLineDistances.argsort()[0]
            self.footHolds[holdColumn] = hold

    def clusterHoldsY(self,holds):
        holdCenters = np.array(list(map(lambda x: x.center, holds)))[:,1]
        x = holdCenters.reshape(-1,1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(x)
        return kmeans.labels_

    def getMissingFootHoldIds(self):
        return np.where(self.footHolds == None)[0]
    
    def getExistingFootHoldIds(self):
        return np.where(self.footHolds != None)[0]
    
    def getExistingFootHolds(self):
        return self.footHolds[np.where(self.footHolds != None)[0]]
    
    def getFootHoldsInColumn(self, j):
        existingHoldIds = self.getExistingFootHoldIds()
        existingHolds = self.getExistingFootHolds()
        return existingHolds[existingHoldIds%self.footBoxLength == j]

    def getFootHoldsInRow(self, i):
        existingHoldIds = self.getExistingFootHoldIds()
        existingHolds = self.getExistingFootHolds()
        return existingHolds[np.floor(existingHoldIds/self.footBoxLength) == i]
    
    def getInterpolatedMissingFootHold(self, missingHoldId):
        missingHoldRow = int(np.floor(missingHoldId/self.footBoxLength))
        missingHoldCol = missingHoldId%self.footBoxLength
        missingHoldName = str(missingHoldId)
        row = self.getFootHoldsInRow(missingHoldRow)
        rowLineParams = self.lineOfBestFit(row)
        colLineParams = self.oddColumnLineParams[missingHoldCol]
        missingHoldCenterX = -(rowLineParams[1]-colLineParams[1])/(rowLineParams[0]-colLineParams[0])
        missingHoldCenterY= (rowLineParams[1]*colLineParams[0]-rowLineParams[0]*colLineParams[1])/(-rowLineParams[0]+colLineParams[0])
        avgHoldSize = self.getAverageHoldSize()
        missingHoldTopLeft = (missingHoldCenterX - avgHoldSize[0]/2, missingHoldCenterY - avgHoldSize[1]/2)
        missingHoldBottomRight = (missingHoldCenterX + avgHoldSize[0]/2, missingHoldCenterY + avgHoldSize[1]/2)
        missingHoldRegion = (missingHoldTopLeft[0],missingHoldTopLeft[1],missingHoldBottomRight[0] ,missingHoldBottomRight[1])
        return Hold(missingHoldName, missingHoldRegion)
        
    def getMissingHandHoldIds(self):
        return np.where(self.handHolds == None)[0]
    
    def getMissingHandHolds(self):
        return self.handHolds[np.where(self.handHolds == None)[0]]
    
    def getExistingHandHoldIds(self):
        return np.where(self.handHolds != None)[0]
    
    def getExistingHandHolds(self):
        return self.handHolds[np.where(self.handHolds != None)[0]]
    
    def getAverageHoldSize(self):
        existingHolds = self.getExistingHandHolds()
        avgHoldWidth = np.average(np.array(list(map(lambda x: x.region[2]-x.region[0], existingHolds))))
        avgHoldHeight = np.average(np.array(list(map(lambda x: x.region[3]-x.region[1], existingHolds))))
        return (avgHoldWidth,avgHoldHeight)
    
    def getHandHoldsInColumn(self, j):
        existingHoldIds = self.getExistingHandHoldIds()
        existingHolds = self.getExistingHandHolds()
        return existingHolds[existingHoldIds%self.handBoxLength == j]

    def getHandHoldsInRow(self, i):
        existingHoldIds = self.getExistingHandHoldIds()
        existingHolds = self.getExistingHandHolds()
        return existingHolds[np.floor(existingHoldIds/self.handBoxLength) == i]

    def getInterpolatedMissingHandHold(self, missingHoldId):
        missingHoldRow = int(np.floor(missingHoldId/self.handBoxLength))
        missingHoldCol = missingHoldId%self.handBoxLength
        missingHoldName = chr(missingHoldCol+ ord('A'))+str(self.handBoxWidth-missingHoldRow)
        col = self.getHandHoldsInColumn(missingHoldCol)
        row = self.getHandHoldsInRow(missingHoldRow)
        rowLineParams = self.lineOfBestFit(row)
        colLineParams = self.lineOfBestFit(col)
        missingHoldCenterX = -(rowLineParams[1]-colLineParams[1])/(rowLineParams[0]-colLineParams[0])
        missingHoldCenterY= (rowLineParams[1]*colLineParams[0]-rowLineParams[0]*colLineParams[1])/(-rowLineParams[0]+colLineParams[0])
        avgHoldSize = self.getAverageHoldSize()
        missingHoldTopLeft = (missingHoldCenterX - avgHoldSize[0]/2, missingHoldCenterY - avgHoldSize[1]/2)
        missingHoldBottomRight = (missingHoldCenterX + avgHoldSize[0]/2, missingHoldCenterY + avgHoldSize[1]/2)
        missingHoldRegion = (missingHoldTopLeft[0],missingHoldTopLeft[1],missingHoldBottomRight[0] ,missingHoldBottomRight[1])
        return Hold(missingHoldName, missingHoldRegion)
    
    def lineOfBestFit(self, holds):
        holdCenters = np.array(list(map(lambda x: list(x.center), holds)))
        regression = LinearRegression().fit(holdCenters[:,0].reshape(-1,1),holdCenters[:,1])
        return [regression.coef_[0], regression.intercept_]

    #def drawHolds(self, holdSubset, color):
    #    for hold in holdSubset:
    #        hold.drawHoldMask(self.draw,color)
    #    self.im.show()


class Hold:
    def __init__(self, name, region):
        shift = ord('A')
        if isinstance(name, int):
            row = str(int(np.floor(name/11))+1)
            col = chr(name%11 +shift)
            self.name = "" + col + row
        else:
            self.name = name
        self.region = region
        self.center = ((region[0]+region[2])/2,(region[1]+region[3])/2)

    def isHand(self):
        if len(self.name) == 1:
            return False 
        else:
            return True   
    
    def getGridIndex(self):
        if self.isHand():
            shift = ord('a')
            col = (ord((self.name[0]).lower())-shift)
            row = 18 - int(self.name[1::])
            return (row,col)
        return (-1,-1)

    def getHoldId(self):
        tempGridIndex = self.getGridIndex()
        return tempGridIndex[0]*11 + tempGridIndex[1]
    
    def drawHoldMask(self,draw,color):
        x1 = int(self.region[0])
        x2 = int(self.region[2])
        y1 = int(self.region[1])
        y2 = int(self.region[3])
        draw.rectangle((x1,y1,x2,y2),fill=color[0])
        draw.circle(self.center, fill=color[1], radius=2)

    def drawHold(self, draw, color):
        x1 = int(self.region[0])
        x2 = int(self.region[2])
        y1 = int(self.region[1])
        y2 = int(self.region[3])
        draw.rectangle((x1,y1,x2,y2),outline=color[0])
        draw.circle(self.center, fill=color[1], radius=2)

    def distanceToHoldLine(self, line):
        p = self.center
        d = np.abs(-line[0]*p[0] + p[1] - line[1])/np.sqrt(line[0]**2 + 1)
        return d  
    

class Person:
    def __init__(self, region, joints):
        self.region = region 
        self.joints = joints

    def getCenter(self):
        personRegion = self.region
        return np.array([personRegion[0]+0.5*(personRegion[2]-personRegion[0]),personRegion[1]+0.5*(personRegion[3]-personRegion[1])])

    def getLimbs(self):
        limbs = np.zeros((8,2))
        limbs[LBICEP,:] = self.getLeftBicep()
        limbs[LFOREARM,:] = self.getLeftForearm()
        limbs[RBICEP,:] = self.getRightBicep()
        limbs[RFOREARM,:] = self.getRightForearm()
        limbs[LQUAD,:] = self.getLeftQuad()
        limbs[LCALF,:] = self.getLeftCalf()
        limbs[RQUAD,:] = self.getRightQuad()
        limbs[RCALF,:] = self.getRightCalf()
        return limbs

    def outlierJoints(self):
        relevantJoints = self.joints[5:17,:]
        outliers = list(filter(lambda x: (x[0] < self.region[0] or x[0] > self.region[2]) or (x[1] < self.region[1] or x[1] > self.region[3]), relevantJoints))
        if len(outliers) > 0:
            return True 
        else:
            return False

    def getLeftForearm(self):
        leftElbow = self.joints[7,:]
        leftWrist = self.joints[9,:]
        return leftWrist - leftElbow

    def getLeftBicep(self):
        leftShoulder = self.joints[5,:]
        leftElbow = self.joints[7,:]
        return leftElbow - leftShoulder

    def getRightForearm(self):
        rightElbow = self.joints[8,:]     
        rightWrist = self.joints[10,:]
        return rightWrist - rightElbow

    def getRightBicep(self):
        rightShoulder = self.joints[6,:]
        rightElbow = self.joints[8,:]     
        return rightElbow - rightShoulder

    def getLeftQuad(self):
        leftHip = self.joints[11,:]
        leftKnee = self.joints[13,:]
        return leftKnee - leftHip

    def getLeftCalf(self):
        leftKnee = self.joints[13,:]
        leftAnkle = self.joints[15,:]
        return leftAnkle - leftKnee

    def getRightQuad(self):
        rightHip = self.joints[12,:]    
        rightKnee = self.joints[14,:]
        return rightKnee - rightHip

    def getRightCalf(self):  
        rightKnee = self.joints[14,:]        
        rightAnkle = self.joints[16,:]
        return rightAnkle-rightKnee

    """
    #Function incomplete dont use
    def resizeLimb(self, limb, avg):
        if limb == LBICEP:
            tempLimb = self.getLeftBicep()
            d = np.linalg.norm(tempLimb)
            self.joints[7,:] = self.joints[7,:] - tempLimb/d*(d-avg)  
        if limb == LBICEP:
            tempLimb = self.getLeftBicep()
            d = np.linalg.norm(tempLimb)
            self.joints[7,:] = self.joints[7,:] - tempLimb/d*(d-avg)  
        if limb == LBICEP:
            tempLimb = self.getLeftBicep()
            d = np.linalg.norm(tempLimb)
            self.joints[7,:] = self.joints[7,:] - tempLimb/d*(d-avg)  
        if limb == LBICEP:
            tempLimb = self.getLeftBicep()
            d = np.linalg.norm(tempLimb)
            self.joints[7,:] = self.joints[7,:] - tempLimb/d*(d-avg)  
        if limb == LBICEP:
            tempLimb = self.getLeftBicep()
            d = np.linalg.norm(tempLimb)
            self.joints[7,:] = self.joints[7,:] - tempLimb/d*(d-avg)  
        if limb == LBICEP:
            tempLimb = self.getLeftBicep()
            d = np.linalg.norm(tempLimb)
            self.joints[7,:] = self.joints[7,:] - tempLimb/d*(d-avg)  
        if limb == LBICEP:
            tempLimb = self.getLeftBicep()
            d = np.linalg.norm(tempLimb)
            self.joints[7,:] = self.joints[7,:] - tempLimb/d*(d-avg)  
        if limb == LBICEP:
            tempLimb = self.getLeftBicep()
            d = np.linalg.norm(tempLimb)
            self.joints[7,:] = self.joints[7,:] - tempLimb/d*(d-avg)  
    """