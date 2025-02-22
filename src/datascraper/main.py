import videos as mcv

"""
The datascraper module can be used to extract moonboard climbing move sequences and poses 
from a video. This file serves as an example of how the general data extraction process works.

This file is the first major step in the remaining portion of this project, where I plan to 
train a neural network to solve moonboard climbing problems with Youtube videos.
"""

# Create an unprocessed video object
video = mcv.Video()

# Load video from sample directory specified by source_path
source_path = "./../../datasets/climbing_videos/mp4/2024_samples/"

# Video names in the source_path
samples=["Birthday cake trail mix v4⧸6b+： moonboard 2024 benchmark [3OSjPqTDS7Q]",
         "Funny thing v4⧸6b+： moonboard 2024 benchmark [VHheANjHfIo]",
         "Everything is 6b+ v4⧸6b+： moonboard 2024 benchmark [KofCyD94t04]"
         ]
source = samples[2]
# Video extension
ext = ".webm"

# Start capturing video at start_time
start_time = 0
# Stop capturing video at end_time
end_time = 1000
# Save video to save_path after processing
save_path = "./../../datasets/climbing_videos/hdf5/2024_samples/"
# Uncomment the following two lines to process new videos. The videos in the samples 
# list above have already been processed and saved to save_path, so no need to uncomment for these examples.
#video.load_from_MP4(source_path, source, ext, "2024", start_time,end_time)
#video.save_to_HDF5(save_path+source+".hdf5")

# Load processed video from HDF5 file
print("Loading from HDF")
video.load_from_HDF5(save_path+source+".hdf5")

# Remove frames that have been incorrectly labelled by the key-point pose model
video.remove_outlier_frames()
p_video = mcv.ProcessedVideo()
# Set the climbing holds for the processed video from the unprocessed video.
p_video.set_board(video)

# Batch size groups the original video frames into groups of 10 and averages them 
# to produce a new video. Typically a batch_size of 10 works well to properly extract
# the move sequences. Smaller batch sizes lead to more noise in the joint time series.
batch_size = 10
p_video.set_frames(video, batch_size)
# Analyze the joint movement from the processed video, and extract the climbing move sequence
p_video.set_moves_from_time_series()
# Output the climbing move sequence
p_video.print_move_sequence()
# Playback the processed video and corresponding hand time series
p_video.play()
