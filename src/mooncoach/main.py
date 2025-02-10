import videos as mcv

video = mcv.Video()
#source = "alien_v4"
#source = "everythingis6b+_v4"
#source="birthdaycaketrailmix_v4"
source="moongirl_v4"


# LOAD VIDEO FROM MP4 AND SAVE AS HDF5
#startTime = 0 
#endTime = 1 #variable not used
#video.loadFromMP4("./../../datasets/climbing_videos/mp4/"+source+".mp4", startTime, endTime)
#video.play()
#video.saveToHDF5("./../../datasets/climbing_videos/hdf5/"+source+".hdf5")


# LOAD VIDEO FROM HDF5
video.loadFromHDF5("./../../datasets/climbing_videos/hdf5/"+source+".hdf5")
video.removeOutlierFrames()
pvideo = mcv.Pvideo()
pvideo.setBoard(video)
pvideo.setFrames(video,10)
pvideo.play()
