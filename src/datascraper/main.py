import videos as mcv

"""
The datascraper module can be used to extract moonboard climbing move sequences and poses 
from a video. This file serves as an example of how the general data extraction process works.
"""

video = mcv.Video() # unprocessed video instance
# Video file names in the source_dir: It's essential that the "video file name" contains the "climb problem name" that exists in
# one of the json datasets/moonboard_sets to properly process data sets since the computer vision model can't identiy valid climb holds from the video alone.
source_dir = "./../../datasets/climbing_videos/mp4/2024_samples/"
save_dir = "./../../datasets/climbing_videos/hdf5/2024_samples/"
samples=["Birthday cake trail mix v4⧸6b+： moonboard 2024 benchmark [3OSjPqTDS7Q]",
         "Funny thing v4⧸6b+： moonboard 2024 benchmark [VHheANjHfIo]",
         "Everything is 6b+ v4⧸6b+： moonboard 2024 benchmark [KofCyD94t04]"
         ]
source_name = samples[2]
ext = ".webm"
# Uncomment the following lines to process new videos. The videos in the samples 
# list above have already been processed and saved to save_path, so no need to uncomment for these examples.
# If Video.load_from_MP4 is unable to add frames, yolo11m_finetuned.pt may need to be further trained on a larger hold dataset.
# models/yolo11m_finetuned.pt is currently trained on datasets/hold_image_datasets/2024_shortuglybeta/train images.
start_time = 0
end_time = 1
video.load_from_MP4(source_dir, source_name, ext, "2024", start_time,end_time)
video.remove_outlier_frames(False)
video.save_to_HDF5(save_dir+source_name+".hdf5")


# Load processed video from HDF5 file
print("Loading from HDF")
video.load_from_HDF5(save_dir+source_name+".hdf5")
# Batch size groups the original video frames into groups of 10 and averages them 
# to produce a new video: larger batch sizes produce choppier videos. 
# Typically a batch_size of 5-10 works well to properly extract
# the move sequences. Smaller batch sizes lead to more noise in the joint time series.
batch_size = 5
moves_save_dir = save_dir + "move_sequences/"
p_video = mcv.ProcessedVideo()
p_video.set_frames(video, 5)
p_video.print_move_sequence()
p_video.save_moves_to_HDF5(moves_save_dir,source_name)
# Playback the processed video and corresponding hand time series
p_video.play()
