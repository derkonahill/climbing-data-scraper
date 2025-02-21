import videos as mcv
import sys
import json

video = mcv.Video()


#source="everythingis6b+"
#source="moongirl"
#source = "alien"
#source="funnything"
#source="placesinbetween"


# LOAD VIDEO FROM MP4 AND SAVE AS HDF5
#climbName = "Funny thing v4/6b+: moonboard 2024 benchmark"
source = sys.argv[1].split('/')[7::-1][0].split('.')[0]
#source = "Admiral snackbar v7⧸7a+： moonboard 2024 benchmark [_2NOYKbHSU8]"

year="2024"
start_time = 0
end_time = 1000
print(source)
with open("./../../datasets/climbing_videos/hdf5/2024_shortuglybeta/" + source + '.json', 'w') as fp:
    sample = {'name': source, 'finished': False, 'frames': 0}
    json.dump(sample, fp)    
try:
    #raise ValueError('A very specific bad thing happened.')
    video.load_from_MP4(source, "2024",start_time,end_time)
    video.save_to_HDF5("./../../datasets/climbing_videos/hdf5/2024_shortuglybeta/"+source+".hdf5")
    with open("./../../datasets/climbing_videos/hdf5/2024_shortuglybeta/" + source + '.json', 'w') as fp:
        sample = {'name': source, 'finished': True, 'frames': len(video.frames)}
        json.dump(sample, fp)    
except: 
    print("Error loading " + source)

"""
# LOAD VIDEO FROM HDF5
print("Loading from HDF")
video.load_from_HDF5("./../../datasets/climbing_videos/hdf5/2024_shortuglybeta/"+source+".hdf5")
if video.climb_in_database == True:
    video.remove_outlier_frames()
    p_video = mcv.ProcessedVideo()
    p_video.set_board(video)

    if int(len(video.frames)) > 50:
        batch_size = int(len(video.frames)/50)
    else:
        batch_size = 1

    
    p_video.set_frames(video, batch_size)
    #print(len(p_video.frames))

    #p_video.set_moves_from_time_series()
    p_video.play()
    #p_video.draw_moves_accurate()
"""