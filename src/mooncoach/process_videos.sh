for f in ./../../datasets/climbing_videos/mp4/2024_shortuglybeta/*.webm
do
    python3 "./main.py" "$f"
    wait
done