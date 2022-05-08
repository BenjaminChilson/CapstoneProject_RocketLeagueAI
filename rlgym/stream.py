import os
import subprocess

# need to call stream.py whenever the bot starts up to start the stream
# i can add audio to this in the future if for some reason we want to stream audio

# in order for this to work we need to have ffmpeg installed on the machine
# https://www.gyan.dev/ffmpeg/builds/
ffmpeg_installation = "C://Program Files//ffmpeg//bin"

# we might have to have a monitor hooked up to the machine for this to work, but i'm not really sure.
# i have an extra one i'm not using if we need it
target_frame_rate = 30
screen_resolution = "1280x720"
stream_resolution = "1280x720"

# keep the stream_key off github please. can someone put it in the discord chat for the account we made
ingestion_server = "rtmp://dfw.contribute.live-video.net/app"
stream_key = ""

#i verified that these settings work fine, we may need to tweak them once we test this while the bot is running though

os.chdir(ffmpeg_installation)
subprocess.call("ffmpeg -f gdigrab -s %s -framerate %s -i title=\"Rocket League (64-bit, DX11, Cooked)\" -c:v libx264 -preset fast -pix_fmt yuv420p -s %s -threads 0 -f flv \"%s/%s\"" % (screen_resolution, target_frame_rate, stream_resolution, ingestion_server, stream_key), shell=True)
