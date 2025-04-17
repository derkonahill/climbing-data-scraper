for f in ./../../datasets/climbing_videos/hdf5/2024_shortuglybeta/*.hdf5
do
    python3 "./test.py" "$f"
    wait
done